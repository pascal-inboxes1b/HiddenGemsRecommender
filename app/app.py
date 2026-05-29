import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob


st.set_page_config(page_title="Hidden Gems Stay Recommender", page_icon="🏨", layout="wide")
st.title("🏨 Hidden Gems Stay Recommender")
st.caption("Find overlooked, high-quality stays within major European cities.")

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "Hotel_Reviews.csv"

ASPECTS = {
    "noise": ["quiet", "noisy", "noise", "loud", "soundproof", "peaceful", "calm", "sleep", "street noise"],
    "location": ["location", "central", "center", "walk", "metro", "station", "transport", "far", "near", "distance"],
    "cleanliness": ["clean", "dirty", "spotless", "hygiene", "bathroom", "room clean", "dust", "smell", "stain"],
    "staff": ["staff", "service", "friendly", "helpful", "rude", "reception", "manager", "check in", "check-in"],
    "value": ["value", "price", "expensive", "cheap", "worth", "money", "overpriced", "cost"],
}

QUIET_KEYWORDS = ["quiet", "peaceful", "calm", "serene", "relaxing", "tranquil", "secluded", "sleep well", "good sleep", "not busy"]
CROWDED_KEYWORDS = ["crowded", "busy", "packed", "touristy", "overcrowded", "noisy", "loud", "queue", "long lines", "too many people"]


def split_sentences(text):
    sentences = re.split(r"[.!?]+", str(text))
    return [s.strip() for s in sentences if s.strip()]


def aspect_sentiment(text, keywords):
    sentences = split_sentences(text)
    scores = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in keywords):
            scores.append(TextBlob(sentence).sentiment.polarity)
    if len(scores) == 0:
        return 0.0
    return np.mean(scores)


def contains_any(text, keywords):
    text = str(text).lower()
    return int(any(keyword in text for keyword in keywords))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def compute_weighted_score(df, weights, score_col="hidden_score_base"):
    out = df.copy()
    score = 0
    for feature, weight in weights.items():
        score += weight * out[feature]
    out[score_col] = score
    return out


def recommend(
    df,
    city=None,
    prefer_quiet=True,
    quiet_weight=0.15,
    avoid_crowds=True,
    crowd_weight=0.10,
    distance_mode="balanced",
    distance_weight=0.10,
    top_k=10,
):
    out = df.copy()

    if city is not None:
        out = out[out["city"].astype(str).str.strip().str.lower() == city.strip().lower()].copy()

    if out.empty:
        return out

    score = out["hidden_score_base"].copy()

    if prefer_quiet:
        quiet_match_score = out["quiet_net"]
    else:
        quiet_match_score = 1 - out["quiet_net"]

    calm_quality_score = 0.50 * quiet_match_score + 0.50 * out["noise_sentiment"]
    score += quiet_weight * calm_quality_score

    if avoid_crowds:
        crowd_match_score = 1 - out["crowded_negative_rate"]
    else:
        crowd_match_score = out["crowded_negative_rate"]

    score += crowd_weight * crowd_match_score

    if distance_mode == "central":
        distance_match_score = out["centrality_score"]
    elif distance_mode == "balanced":
        distance_match_score = out["distance_sweet_spot"]
    elif distance_mode == "outer":
        distance_match_score = out["distance_percentile_within_city"]
    else:
        distance_match_score = out["distance_sweet_spot"]

    score += distance_weight * distance_match_score

    out["quiet_match_score"] = quiet_match_score
    out["calm_quality_score"] = calm_quality_score
    out["crowd_match_score"] = crowd_match_score
    out["distance_match_score"] = distance_match_score
    out["final_score"] = score
    out["city_final_percentile"] = out.groupby("city")["final_score"].rank(method="average", pct=True)

    return out.sort_values("final_score", ascending=False).head(top_k)


def explain(row, distance_mode="balanced"):
    reasons = []

    if row["calm_quality_score"] > 0.70:
        reasons.append("strong quiet/calm match")
    if row["noise_sentiment"] > 0.70:
        reasons.append("positive noise/quiet sentiment")
    if row["cleanliness_sentiment"] > 0.70:
        reasons.append("cleanliness is praised")
    if row["staff_sentiment"] > 0.70:
        reasons.append("staff/service is praised")
    if row["location_sentiment"] > 0.70:
        reasons.append("location is reviewed positively")
    if row["value_sentiment"] > 0.70:
        reasons.append("guests mention good value")
    if row["crowd_match_score"] > 0.75:
        reasons.append("matches your crowd preference")

    if distance_mode == "central" and row["centrality_score"] > 0.75:
        reasons.append("close to the city center")
    elif distance_mode == "balanced" and row["distance_sweet_spot"] > 0.80:
        reasons.append("away from the busiest core without being too far out")
    elif distance_mode == "outer" and row["distance_percentile_within_city"] > 0.75:
        reasons.append("farther from the busiest center areas")

    if row["city_final_percentile"] > 0.85:
        reasons.append("high hidden-gem match within this city")

    if not reasons:
        reasons.append("balanced overall profile")

    return ", ".join(reasons)


@st.cache_data(show_spinner=True)
def load_and_prepare_data(path, sample_size=100_000):
    df = pd.read_csv(path)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    df = df[
        [
            "Hotel_Name",
            "Hotel_Address",
            "Reviewer_Score",
            "Total_Number_of_Reviews",
            "Positive_Review",
            "Negative_Review",
            "lat",
            "lng",
            "Province_Name",
        ]
    ].copy()

    df["Positive_Review"] = df["Positive_Review"].fillna("").astype(str).str.strip()
    df["Negative_Review"] = df["Negative_Review"].fillna("").astype(str).str.strip()
    df.dropna(subset=["lat", "lng"], inplace=True)

    df = df.rename(columns={"Province_Name": "Metro_Area"})
    df["city"] = df["Metro_Area"].astype(str).str.strip()

    for aspect, keywords in ASPECTS.items():
        df[f"{aspect}_pos_sentiment"] = df["Positive_Review"].apply(lambda x, kw=keywords: aspect_sentiment(x, kw))
        df[f"{aspect}_neg_sentiment"] = df["Negative_Review"].apply(lambda x, kw=keywords: aspect_sentiment(x, kw))
        df[f"{aspect}_net_sentiment"] = df[f"{aspect}_pos_sentiment"] - df[f"{aspect}_neg_sentiment"]

    df["quiet_in_positive"] = df["Positive_Review"].apply(lambda x: contains_any(x, QUIET_KEYWORDS))
    df["quiet_in_negative"] = df["Negative_Review"].apply(lambda x: contains_any(x, QUIET_KEYWORDS))
    df["crowded_in_positive"] = df["Positive_Review"].apply(lambda x: contains_any(x, CROWDED_KEYWORDS))
    df["crowded_in_negative"] = df["Negative_Review"].apply(lambda x: contains_any(x, CROWDED_KEYWORDS))

    df["quiet_net"] = df["quiet_in_positive"] - df["quiet_in_negative"]
    df["crowded_net"] = df["crowded_in_positive"] - df["crowded_in_negative"]

    hotel_stats = df.groupby(["Hotel_Name", "city", "lat", "lng"], as_index=False).agg(
        avg_rating=("Reviewer_Score", "mean"),
        review_count=("Reviewer_Score", "count"),
        dataset_total_reviews=("Total_Number_of_Reviews", "max"),
        noise_sentiment=("noise_net_sentiment", "mean"),
        location_sentiment=("location_net_sentiment", "mean"),
        cleanliness_sentiment=("cleanliness_net_sentiment", "mean"),
        staff_sentiment=("staff_net_sentiment", "mean"),
        value_sentiment=("value_net_sentiment", "mean"),
        quiet_positive_rate=("quiet_in_positive", "mean"),
        quiet_negative_rate=("quiet_in_negative", "mean"),
        crowded_positive_rate=("crowded_in_positive", "mean"),
        crowded_negative_rate=("crowded_in_negative", "mean"),
        quiet_net=("quiet_net", "mean"),
        crowded_net=("crowded_net", "mean"),
    )

    scale_cols = [
        "avg_rating",
        "review_count",
        "dataset_total_reviews",
        "noise_sentiment",
        "location_sentiment",
        "cleanliness_sentiment",
        "staff_sentiment",
        "value_sentiment",
        "quiet_positive_rate",
        "quiet_negative_rate",
        "crowded_positive_rate",
        "crowded_negative_rate",
        "quiet_net",
        "crowded_net",
    ]

    scaler = MinMaxScaler()
    hotel_stats[scale_cols] = scaler.fit_transform(hotel_stats[scale_cols])

    city_centers = pd.DataFrame(
        {
            "city": ["Milan", "Amsterdam", "Barcelona", "London", "Paris", "Vienna"],
            "city_center_lat": [45.4668, 52.3728, 41.3825, 51.5073, 48.8535, 48.2084],
            "city_center_lng": [9.1905, 4.8936, 2.1769, -0.1277, 2.3484, 16.3725],
        }
    )

    hotel_stats["city"] = hotel_stats["city"].astype(str).str.strip()
    city_centers["city"] = city_centers["city"].astype(str).str.strip()
    hotel_stats = hotel_stats.merge(city_centers, on="city", how="left")

    hotel_stats["distance_from_center_km"] = haversine_km(
        hotel_stats["lat"],
        hotel_stats["lng"],
        hotel_stats["city_center_lat"],
        hotel_stats["city_center_lng"],
    )

    hotel_stats["distance_percentile_within_city"] = hotel_stats.groupby("city")["distance_from_center_km"].rank(method="average", pct=True)
    hotel_stats["distance_from_center_pct"] = (100 * hotel_stats["distance_percentile_within_city"]).round(1)
    hotel_stats["distance_sweet_spot"] = 1 - (2 * np.abs(hotel_stats["distance_percentile_within_city"] - 0.5))
    hotel_stats["centrality_score"] = 1 - hotel_stats["distance_percentile_within_city"]

    hotel_stats["log_review_count"] = np.log1p(hotel_stats["dataset_total_reviews"])

    BASE_WEIGHTS = {
        "avg_rating": 0.30,
        "cleanliness_sentiment": 0.15,
        "staff_sentiment": 0.15,
        "location_sentiment": 0.10,
        "value_sentiment": 0.10,
        "noise_sentiment": 0.10,
        "crowded_negative_rate": -0.05,
        "log_review_count": -0.05,
    }

    hotel_stats = compute_weighted_score(hotel_stats, BASE_WEIGHTS, score_col="hidden_score_base")

    hotel_stats["city_hidden_percentile"] = hotel_stats.groupby("city")["hidden_score_base"].rank(method="average", pct=True)

    return hotel_stats


st.sidebar.header("Filters & Preferences")

sample_choice = st.sidebar.selectbox(
    "Rows to process",
    options=["100,000 rows", "200,000 rows", "Full dataset"],
    index=0,
    help="Use a sample while developing. Full dataset may take longer.",
)

if sample_choice == "100,000 rows":
    sample_size = 100_000
elif sample_choice == "200,000 rows":
    sample_size = 200_000
else:
    sample_size = None

with st.spinner("Loading data and building recommendation features..."):
    try:
        hotel_stats = load_and_prepare_data(DATA_PATH, sample_size=sample_size)
    except FileNotFoundError:
        st.error(f"Could not find the dataset at `{DATA_PATH}`.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

city_options = sorted(hotel_stats["city"].dropna().astype(str).unique().tolist())

if len(city_options) == 0:
    st.error("No cities were found.")
    st.stop()

default_city = "Paris" if "Paris" in city_options else city_options[0]
selected_city = st.sidebar.selectbox("City", options=city_options, index=city_options.index(default_city))

st.sidebar.markdown("---")
st.sidebar.subheader("Preference matching")

prefer_quiet = st.sidebar.checkbox("Prefer quiet/calm stays", value=True)
quiet_weight = st.sidebar.slider("Quiet preference strength", 0.00, 0.30, 0.15, 0.01, disabled=not prefer_quiet)

avoid_crowds = st.sidebar.checkbox("Avoid crowded/noisy stays", value=True)
crowd_weight = st.sidebar.slider("Crowd preference strength", 0.00, 0.30, 0.10, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Location preference")

distance_mode_label = st.sidebar.radio("Where do you want to stay?", ["Central", "Balanced", "Farther from center"], index=1)

distance_mode_map = {
    "Central": "central",
    "Balanced": "balanced",
    "Farther from center": "outer",
}
distance_mode = distance_mode_map[distance_mode_label]

distance_weight = st.sidebar.slider("Location preference strength", 0.00, 0.30, 0.10, 0.01)

top_k = st.sidebar.slider("Number of recommendations", 5, 30, 10, 1)

results = recommend(
    hotel_stats,
    city=selected_city,
    prefer_quiet=prefer_quiet,
    quiet_weight=quiet_weight,
    avoid_crowds=avoid_crowds,
    crowd_weight=crowd_weight,
    distance_mode=distance_mode,
    distance_weight=distance_weight,
    top_k=top_k,
)

if results.empty:
    st.warning("No recommendations found for this city.")
    st.stop()

results = results.copy()
results["explanation"] = results.apply(lambda row: explain(row, distance_mode=distance_mode), axis=1)

st.subheader(f"Top hidden stays in {selected_city}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hotels shown", len(results))
c2.metric("Avg final score", f"{results['final_score'].mean():.3f}")
c3.metric("Avg distance", f"{results['distance_from_center_km'].mean():.2f} km")
c4.metric("Avg calm match", f"{results['calm_quality_score'].mean():.3f}")

display_df = results[
    [
        "Hotel_Name",
        "final_score",
        "hidden_score_base",
        "city_final_percentile",
        "avg_rating",
        "calm_quality_score",
        "crowd_match_score",
        "distance_match_score",
        "noise_sentiment",
        "cleanliness_sentiment",
        "staff_sentiment",
        "location_sentiment",
        "value_sentiment",
        "distance_from_center_km",
        "distance_from_center_pct",
        "explanation",
    ]
].copy()

display_df = display_df.rename(
    columns={
        "Hotel_Name": "Hotel",
        "final_score": "Final score",
        "hidden_score_base": "Base hidden score",
        "city_final_percentile": "Final percentile in city",
        "avg_rating": "Rating",
        "calm_quality_score": "Calm match",
        "crowd_match_score": "Crowd match",
        "distance_match_score": "Distance match",
        "noise_sentiment": "Noise sentiment",
        "cleanliness_sentiment": "Cleanliness sentiment",
        "staff_sentiment": "Staff sentiment",
        "location_sentiment": "Location sentiment",
        "value_sentiment": "Value sentiment",
        "distance_from_center_km": "Distance from center (km)",
        "distance_from_center_pct": "Distance percentile in city",
        "explanation": "Why recommended",
    }
)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.subheader("Map")

fig = px.scatter_mapbox(
    results,
    lat="lat",
    lon="lng",
    hover_name="Hotel_Name",
    hover_data={
        "city": True,
        "final_score": ":.3f",
        "hidden_score_base": ":.3f",
        "calm_quality_score": ":.3f",
        "crowd_match_score": ":.3f",
        "distance_match_score": ":.3f",
        "distance_from_center_km": ":.2f",
        "city_final_percentile": ":.3f",
        "lat": False,
        "lng": False,
    },
    color="final_score",
    size="city_final_percentile",
    color_continuous_scale="Viridis",
    zoom=10,
    height=650,
    title=f"Hidden stays in {selected_city}",
)

fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=50, b=0))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Methodology"):
    st.markdown(
        """
        **Business logic summary**

        - The app identifies hidden stays within major European cities, not hidden cities.
        - The base hidden-gem score uses expert-defined weights over rating, aspect-based sentiment, and popularity.
        - Aspect-based sentiment extracts guest sentiment around noise, cleanliness, staff, location, and value.
        - Positive and negative review fields are analyzed separately.
        - Haversine distance estimates how far each hotel is from the city center.
        - Distance is converted into within-city percentile features.
        - User preferences are applied at recommendation time, not baked into the base score.
        - The checkbox + slider pattern means checkbox = whether the preference direction matters, and slider = how strongly it affects ranking.
        """
    )
