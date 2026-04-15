import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Hidden Gems Stay Recommender",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hidden Gems Stay Recommender")
st.caption("Find overlooked, high-quality hotels within major European tourist cities.")


# ----------------------------
# Config
# ----------------------------
DATA_PATH = "../data/Hotel_Reviews.csv"


# ----------------------------
# Helper functions
# ----------------------------

def polarity(text):
    text = str(text).strip()
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity


QUIET_KEYWORDS = [
    "quiet", "peaceful", "calm", "serene", "relaxing",
    "tranquil", "secluded", "sleep well", "good sleep", "not busy"
]

CROWDED_KEYWORDS = [
    "crowded", "busy", "packed", "touristy", "overcrowded",
    "noisy", "loud", "queue", "long lines", "too many people"
]


def contains_any(text, keywords):
    text = str(text).lower()
    return int(any(k in text for k in keywords))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def explain(row, distance_mode="balanced"):
    reasons = []

    if row["quiet_positive_rate"] > 0.60:
        reasons.append("guests often praise how quiet it is")

    if row["quiet_negative_rate"] < 0.20:
        reasons.append("quietness is rarely mentioned as a downside")

    if row["crowded_negative_rate"] < 0.25:
        reasons.append("fewer complaints about crowds and noise")

    if row["avg_rating"] > 0.70:
        reasons.append("strong review scores")

    if distance_mode == "central" and row["centrality_score"] > 0.75:
        reasons.append("close to the city center")
    elif distance_mode == "balanced" and row["distance_sweet_spot"] > 0.80:
        reasons.append("away from the busiest core without being too far out")
    elif distance_mode == "outer" and row["distance_percentile_within_city"] > 0.75:
        reasons.append("farther from the busiest center areas")

    if row["city_hidden_percentile"] > 0.85:
        reasons.append("stands out as an overlooked option within its city")

    if not reasons:
        reasons.append("balanced overall profile")

    return ", ".join(reasons)


def recommend(
    df,
    city=None,
    quiet_weight=0.15,
    crowd_weight=0.15,
    distance_mode="balanced",   # "central", "balanced", "outer"
    distance_weight=0.05,
    top_k=10
):
    out = df.copy()

    if city is not None:
        out = out[
            out["city"].astype(str).str.strip().str.lower() == city.strip().lower()
        ].copy()

    score = out["hidden_score_base"].copy()

    score += quiet_weight * out["quiet_net"]
    score -= crowd_weight * out["crowded_negative_rate"]

    if distance_mode == "central":
        score += distance_weight * out["centrality_score"]
    elif distance_mode == "balanced":
        score += distance_weight * out["distance_sweet_spot"]
    elif distance_mode == "outer":
        score += distance_weight * out["distance_percentile_within_city"]

    out["final_score"] = score
    out["explanation"] = out.apply(lambda row: explain(row, distance_mode=distance_mode), axis=1)

    return out.sort_values("final_score", ascending=False).head(top_k)


# ----------------------------
# Data preparation
# ----------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data(path):
    df = pd.read_csv(path)

    # Optional development sample:
    df = df.sample(100000, random_state=42)

    df = df[[
        "Hotel_Name",
        "Hotel_Address",
        "Reviewer_Score",
        "Total_Number_of_Reviews",
        "Positive_Review",
        "Negative_Review",
        "lat",
        "lng",
        "Province_Name"
    ]].copy()

    df["Positive_Review"] = df["Positive_Review"].fillna("").astype(str).str.strip()
    df["Negative_Review"] = df["Negative_Review"].fillna("").astype(str).str.strip()

    df.dropna(subset=["lat", "lng"], inplace=True)

    df["city"] = df["Province_Name"].astype("string").str.strip()

    # Sentiment features
    df["pos_sentiment"] = df["Positive_Review"].apply(polarity)
    df["neg_sentiment"] = df["Negative_Review"].apply(polarity)
    df["net_sentiment"] = df["pos_sentiment"] - df["neg_sentiment"]

    # Directional keyword features
    df["quiet_in_positive"] = df["Positive_Review"].apply(lambda x: contains_any(x, QUIET_KEYWORDS))
    df["quiet_in_negative"] = df["Negative_Review"].apply(lambda x: contains_any(x, QUIET_KEYWORDS))

    df["crowded_in_positive"] = df["Positive_Review"].apply(lambda x: contains_any(x, CROWDED_KEYWORDS))
    df["crowded_in_negative"] = df["Negative_Review"].apply(lambda x: contains_any(x, CROWDED_KEYWORDS))

    df["quiet_net"] = df["quiet_in_positive"] - df["quiet_in_negative"]
    df["crowded_net"] = df["crowded_in_positive"] - df["crowded_in_negative"]

    # Hotel-level aggregation
    hotel_stats = df.groupby(
        ["Hotel_Name", "city", "lat", "lng"],
        as_index=False
    ).agg(
        avg_rating=("Reviewer_Score", "mean"),
        review_count=("Reviewer_Score", "count"),
        dataset_total_reviews=("Total_Number_of_Reviews", "max"),
        pos_sentiment=("pos_sentiment", "mean"),
        neg_sentiment=("neg_sentiment", "mean"),
        net_sentiment=("net_sentiment", "mean"),
        quiet_positive_rate=("quiet_in_positive", "mean"),
        quiet_negative_rate=("quiet_in_negative", "mean"),
        crowded_positive_rate=("crowded_in_positive", "mean"),
        crowded_negative_rate=("crowded_in_negative", "mean"),
        quiet_net=("quiet_net", "mean"),
        crowded_net=("crowded_net", "mean"),
    )

    # Scale model features
    scale_cols = [
        "avg_rating",
        "review_count",
        "dataset_total_reviews",
        "pos_sentiment",
        "neg_sentiment",
        "net_sentiment",
        "quiet_positive_rate",
        "quiet_negative_rate",
        "crowded_positive_rate",
        "crowded_negative_rate",
        "quiet_net",
        "crowded_net",
    ]

    scaler = MinMaxScaler()
    hotel_stats[scale_cols] = scaler.fit_transform(hotel_stats[scale_cols])

    # City centers
    city_centers = pd.DataFrame({
        "city": ["Milan", "Amsterdam", "Barcelona", "London", "Paris", "Vienna"],
        "city_center_lat": [45.4668, 52.3728, 41.3825, 51.5073, 48.8535, 48.2084],
        "city_center_lng": [9.1905, 4.8936, 2.1769, -0.1277, 2.3484, 16.3725],
    })

    hotel_stats["city"] = hotel_stats["city"].astype(str).str.strip()
    city_centers["city"] = city_centers["city"].astype(str).str.strip()
    hotel_stats = hotel_stats.merge(city_centers, on="city", how="left")

    hotel_stats["distance_from_center_km"] = haversine_km(
        hotel_stats["lat"],
        hotel_stats["lng"],
        hotel_stats["city_center_lat"],
        hotel_stats["city_center_lng"]
    )

    hotel_stats["distance_percentile_within_city"] = (
        hotel_stats.groupby("city")["distance_from_center_km"]
        .rank(method="average", pct=True)
    )

    hotel_stats["distance_from_center_pct"] = (
        100 * hotel_stats["distance_percentile_within_city"]
    ).round(1)

    hotel_stats["distance_sweet_spot"] = (
        1 - (2 * np.abs(hotel_stats["distance_percentile_within_city"] - 0.5))
    )

    hotel_stats["centrality_score"] = 1 - hotel_stats["distance_percentile_within_city"]

    # Base hidden gem score: no user distance preference baked in
    hotel_stats["hidden_score_base"] = (
        0.30 * hotel_stats["avg_rating"]
        + 0.20 * hotel_stats["net_sentiment"]
        + 0.20 * hotel_stats["quiet_net"]
        - 0.15 * hotel_stats["crowded_negative_rate"]
        - 0.10 * hotel_stats["dataset_total_reviews"]
    )

    hotel_stats["city_hidden_percentile"] = (
        hotel_stats.groupby("city")["hidden_score_base"]
        .rank(method="average", pct=True)
    )

    return hotel_stats


# ----------------------------
# Load data
# ----------------------------
try:
    hotel_stats = load_and_prepare_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find the dataset at `{DATA_PATH}`. "
        "Make sure `Hotel_Reviews.csv` is inside the `data/` folder."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Filters & preferences")

city_options = sorted(hotel_stats["city"].dropna().astype(str).unique().tolist())
default_city = "Paris" if "Paris" in city_options else city_options[0]

selected_city = st.sidebar.selectbox(
    "City",
    options=city_options,
    index=city_options.index(default_city)
)

quiet_weight = st.sidebar.slider(
    "Prefer quiet",
    min_value=0.0,
    max_value=0.30,
    value=0.15,
    step=0.01
)

crowd_weight = st.sidebar.slider(
    "Avoid crowds",
    min_value=0.0,
    max_value=0.30,
    value=0.15,
    step=0.01
)

distance_mode = st.sidebar.selectbox(
    "Location preference",
    options=["central", "balanced", "outer"],
    index=1
)

distance_weight = st.sidebar.slider(
    "How much location matters",
    min_value=0.0,
    max_value=0.20,
    value=0.05,
    step=0.01
)

top_k = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=30,
    value=10,
    step=1
)


# ----------------------------
# Generate recommendations
# ----------------------------
results = recommend(
    hotel_stats,
    city=selected_city,
    quiet_weight=quiet_weight,
    crowd_weight=crowd_weight,
    distance_mode=distance_mode,
    distance_weight=distance_weight,
    top_k=top_k
)


# ----------------------------
# Summary metrics
# ----------------------------
st.subheader(f"Top hidden stays in {selected_city}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hotels shown", len(results))
c2.metric("Avg final score", f"{results['final_score'].mean():.3f}")
c3.metric("Avg quiet score", f"{results['quiet_net'].mean():.3f}")
c4.metric("Avg distance from center", f"{results['distance_from_center_km'].mean():.2f} km")


# ----------------------------
# Recommendations table
# ----------------------------
display_df = results[[
    "Hotel_Name",
    "final_score",
    "avg_rating",
    "quiet_positive_rate",
    "crowded_negative_rate",
    "distance_from_center_km",
    "distance_from_center_pct",
    "city_hidden_percentile",
    "explanation"
]].copy()

display_df = display_df.rename(columns={
    "Hotel_Name": "Hotel",
    "final_score": "Final score",
    "avg_rating": "Rating",
    "quiet_positive_rate": "Quiet praise rate",
    "crowded_negative_rate": "Crowd complaint rate",
    "distance_from_center_km": "Distance from center (km)",
    "distance_from_center_pct": "Distance percentile in city",
    "city_hidden_percentile": "Hidden-gem percentile in city",
    "explanation": "Why it was recommended"
})

st.dataframe(display_df, use_container_width=True, hide_index=True)


# ----------------------------
# Map
# ----------------------------
st.subheader("Map")

fig = px.scatter_mapbox(
    results,
    lat="lat",
    lon="lng",
    hover_name="Hotel_Name",
    hover_data={
        "city": True,
        "final_score": ':.3f',
        "avg_rating": ':.3f',
        "distance_from_center_km": ':.2f',
        "distance_from_center_pct": ':.1f',
        "city_hidden_percentile": ':.3f',
        "lat": False,
        "lng": False,
    },
    color="final_score",
    size="city_hidden_percentile",
    color_continuous_scale="Viridis",
    zoom=10,
    height=650,
    title=f"Hidden stays in {selected_city}"
)

fig.update_layout(
    mapbox_style="open-street-map",
    margin=dict(l=0, r=0, t=50, b=0)
)

st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Top across all covered cities
# ----------------------------
st.subheader("Top 5 hotels from each covered city")

top_per_city = (
    hotel_stats.sort_values("hidden_score_base", ascending=False)
    .groupby("city", as_index=False)
    .head(5)
    .copy()
)

top_per_city["explanation"] = top_per_city.apply(
    lambda row: explain(row, distance_mode="balanced"),
    axis=1
)

st.dataframe(
    top_per_city[[
        "Hotel_Name",
        "city",
        "hidden_score_base",
        "city_hidden_percentile",
        "explanation"
    ]].rename(columns={
        "Hotel_Name": "Hotel",
        "city": "City",
        "hidden_score_base": "Base hidden score",
        "city_hidden_percentile": "Hidden-gem percentile in city",
        "explanation": "Why it stands out"
    }),
    use_container_width=True,
    hide_index=True
)
