from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "global_single_output_hist_gradient_boosting.joblib"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "ethiopia_food_prices_cleaned.csv"
COMMODITIES = [
    "livestock_goat",
    "livestock_sheep",
    "maize",
    "pasta",
    "sorghum",
    "teff_fao",
    "wheat",
]
COMMODITY_LABELS = {
    "Teff (ETB/kg)": "teff_fao",
    "Maize (ETB/kg)": "maize",
    "Wheat (ETB/kg)": "wheat",
    "Sorghum (ETB/kg)": "sorghum",
    "Pasta (ETB/kg)": "pasta",
    "Goat (ETB/head)": "livestock_goat",
    "Sheep (ETB/head)": "livestock_sheep",
}

st.set_page_config(page_title="Food Price Forecaster", page_icon="📈", layout="wide")


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH, parse_dates=["price_date"])
    data = data.sort_values(["geo_id", "price_date"]).reset_index(drop=True)
    return data


@st.cache_data
def build_feature_panel(data: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for commodity in COMMODITIES:
        part = data[
            [
                "geo_id",
                "mkt_name",
                "adm1_name",
                "lat",
                "lon",
                "price_date",
                "month",
                "c_food_price_index",
                f"c_{commodity}_standardized",
                f"{commodity}_standardized",
                commodity,
                f"{commodity}_outlier_suspect",
                f"trust_{commodity}",
            ]
        ].copy()
        part = part.rename(
            columns={
                f"c_{commodity}_standardized": "price",
                f"{commodity}_standardized": "direct_price",
                commodity: "raw_direct_price",
                f"{commodity}_outlier_suspect": "direct_outlier_suspect",
                f"trust_{commodity}": "trust",
            }
        )
        part["commodity"] = commodity
        parts.append(part)

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["geo_id", "commodity", "price_date"]).reset_index(drop=True)
    panel["direct_outlier_suspect"] = panel["direct_outlier_suspect"].fillna(False).astype(bool)
    panel["is_direct_survey"] = panel["raw_direct_price"].notna().astype("int8")
    panel["direct_ground_truth"] = panel["direct_price"].notna() & ~panel["direct_outlier_suspect"]

    groups = panel.groupby(["geo_id", "commodity"], sort=False, observed=True)
    for lag in [1, 2, 3, 6, 12]:
        panel[f"lag_{lag}"] = groups["price"].shift(lag)
    panel["log_price"] = np.log1p(panel["price"].where(panel["price"] > -1))
    for window in [3, 6]:
        panel[f"roll_mean_{window}"] = groups["price"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        panel[f"roll_std_{window}"] = groups["price"].transform(
            lambda values: values.rolling(window, min_periods=window).std(ddof=0)
        )
    panel["pct_change_1m"] = groups["price"].pct_change(fill_method=None)
    panel["pct_change_12m"] = groups["price"].pct_change(12, fill_method=None)

    macro_change = data[["geo_id", "price_date", "c_food_price_index"]].copy()
    macro_change = macro_change.sort_values(["geo_id", "price_date"])
    macro_change["food_price_index_pct_change_1m"] = macro_change.groupby(
        "geo_id", sort=False
    )["c_food_price_index"].pct_change(fill_method=None)
    panel = panel.merge(
        macro_change[["geo_id", "price_date", "food_price_index_pct_change_1m"]],
        on=["geo_id", "price_date"],
        how="left",
        validate="many_to_one",
    )

    panel["next_price"] = groups["price"].shift(-1)
    panel["next_date"] = groups["price_date"].shift(-1)
    panel["next_direct_price"] = groups["direct_price"].shift(-1)
    panel["next_direct_ground_truth"] = groups["direct_ground_truth"].shift(-1).fillna(False).astype(bool)
    return panel


def format_price(value: float, unit: str) -> str:
    return f"{value:,.2f} {unit}"


@st.cache_data
def add_model_categories(
    row: pd.DataFrame, commodity_levels: tuple[str, ...], region_levels: tuple[str, ...]
) -> pd.DataFrame:
    ready = row.copy()
    ready["commodity"] = pd.Categorical(ready["commodity"], categories=list(commodity_levels))
    ready["adm1_name"] = pd.Categorical(ready["adm1_name"], categories=list(region_levels))
    return ready


st.title("Ethiopia Food Price Forecaster")
st.caption("Forecast next-month prices for selected markets and commodities.")

try:
    bundle = load_model_bundle()
    dataset = load_dataset()
    panel = build_feature_panel(dataset)
except Exception as exc:
    st.error("The forecasting files could not be loaded. Check that the model and cleaned price data are deployed.")
    st.stop()

regions = sorted(dataset["adm1_name"].dropna().unique().tolist())
if not regions:
    st.error("No regions are available in the price dataset.")
    st.stop()
def_region = regions.index("Addis Ababa") if "Addis Ababa" in regions else 0

control_region, control_market, control_commodity = st.columns(3)
with control_region:
    selected_region = st.selectbox("Select Region", regions, index=def_region)

markets = sorted(
    dataset.loc[dataset["adm1_name"].eq(selected_region), "mkt_name"].dropna().unique().tolist()
)
if not markets:
    st.warning("No markets are listed for this region.")
    st.stop()
with control_market:
    selected_market = st.selectbox("Select Market", markets)
with control_commodity:
    selected_label = st.selectbox("Select Commodity", list(COMMODITY_LABELS))
selected_commodity = COMMODITY_LABELS[selected_label]
unit = selected_label.split("(")[1].rstrip(")")

market_rows = panel.loc[
    panel["adm1_name"].eq(selected_region)
    & panel["mkt_name"].eq(selected_market)
    & panel["commodity"].eq(selected_commodity)
].sort_values("price_date")
if market_rows.empty:
    st.warning("No price history is available for this selection.")
    st.stop()

latest_date = market_rows["price_date"].max()
latest_label = f"Latest available ({latest_date:%b %Y})"
latest_key = "__latest__"
last_test_date = latest_date - pd.DateOffset(months=1)
historical_dates = list(
    pd.date_range("2024-01-01", last_test_date, freq="MS")
) if last_test_date >= pd.Timestamp("2024-01-01") else []
date_options = [latest_label] + [f"{date:%b %Y}" for date in historical_dates]
date_lookup = {latest_label: latest_date}
date_lookup.update({f"{date:%b %Y}": date for date in historical_dates})
selected_period = st.selectbox("Forecast from", date_options)
feature_date = date_lookup[selected_period]

selected_row = market_rows.loc[market_rows["price_date"].eq(feature_date)]
if selected_row.empty:
    st.warning("This market has no price record for the selected month.")
    st.stop()
row = selected_row.iloc[[0]].copy()
feature_names = bundle["features"]
missing_features = [name for name in feature_names if name not in row.columns]
if missing_features:
    st.error("The selected model needs features that are not available in the prepared data.")
    st.stop()
if row[feature_names].isna().any(axis=None):
    st.warning("There is not enough complete price history to forecast from this month.")
    st.stop()

region_levels = tuple(sorted(dataset["adm1_name"].dropna().unique().tolist()))
commodity_levels = tuple(bundle.get("commodities", COMMODITIES))
model_input = add_model_categories(row[feature_names], commodity_levels, region_levels)
predicted_log_diff = float(bundle["model"].predict(model_input)[0])
current_price = float(row["price"].iloc[0])
predicted_price = float(np.expm1(np.log1p(current_price) + predicted_log_diff))
price_difference = predicted_price - current_price
percent_change = (price_difference / current_price * 100) if current_price else np.nan
forecast_date = feature_date + pd.DateOffset(months=1)

metric_current, metric_prediction, metric_change = st.columns(3)
metric_current.metric(
    f"Current Closing Estimate · {feature_date:%b %Y}",
    format_price(current_price, unit),
)
metric_prediction.metric(
    f"Predicted Next-Month Price · {forecast_date:%b %Y}",
    format_price(predicted_price, unit),
)
metric_change.metric(
    "Expected Change (%)",
    f"{percent_change:+.2f}%" if np.isfinite(percent_change) else "—",
    delta=f"{price_difference:+,.2f} {unit}",
)

if not bool(row["direct_ground_truth"].iloc[0]):
    last_direct_date = market_rows.loc[
        market_rows["direct_ground_truth"], "price_date"
    ].max()
    if pd.isna(last_direct_date):
        source_note = (
            "No direct survey is available for this market and commodity in the dataset. "
            "The forecast uses the closing-estimate series."
        )
    else:
        source_note = (
            f"No direct survey was recorded in {feature_date:%b %Y}; the most recent "
            f"for this selection was {last_direct_date:%b %Y}. "
            "The forecast uses the closing-estimate series."
        )
    st.caption(source_note)

next_rows = market_rows.loc[market_rows["price_date"].eq(forecast_date)]
if not next_rows.empty:
    actual_row = next_rows.iloc[0]
    actual_close = actual_row["price"]
    actual_direct = actual_row["direct_price"]
    if pd.notna(actual_close):
        st.info(f"Recorded closing estimate for {forecast_date:%b %Y}: {format_price(float(actual_close), unit)}.")
    if bool(actual_row["direct_ground_truth"]) and pd.notna(actual_direct):
        st.success(f"Non-outlier direct survey for {forecast_date:%b %Y}: {format_price(float(actual_direct), unit)}.")

history = market_rows.loc[market_rows["price_date"].le(feature_date)].tail(24)
chart = go.Figure()
chart.add_trace(
    go.Scatter(
        x=history["price_date"],
        y=history["price"],
        mode="lines+markers",
        name="Closing estimate",
        line={"color": "#2563eb", "width": 2},
    )
)
chart.add_trace(
    go.Scatter(
        x=[forecast_date],
        y=[predicted_price],
        mode="markers",
        name="Forecast",
        marker={"color": "#dc2626", "size": 12, "symbol": "diamond"},
    )
)
if not next_rows.empty and pd.notna(next_rows.iloc[0]["price"]):
    chart.add_trace(
        go.Scatter(
            x=[forecast_date],
            y=[float(next_rows.iloc[0]["price"])],
            mode="markers",
            name="Recorded closing estimate",
            marker={"color": "#16a34a", "size": 10, "symbol": "circle-open"},
        )
    )
chart.update_layout(
    title=f"{selected_label} · {selected_market}",
    xaxis_title="Month",
    yaxis_title=unit,
    hovermode="x unified",
    legend_title_text="",
    margin={"l": 20, "r": 20, "t": 55, "b": 20},
)
st.plotly_chart(chart, width="stretch")
