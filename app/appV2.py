import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
    page_title="Hidden Gems Stay Recommender",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hidden Gems Stay Recommender")
st.caption(
    "Find overlooked, high-quality stays within major European cities using "
    "aspect-based sentiment, crowd signals, and location preferences."
)

DATA_PATH = "../data/Hotel_Reviews.csv"


#def extract_city(address):
#    if pd.isna(address):
#        return None
#
#    parts = [p.strip() for p in str(address).split(",") if p.strip()]
#
#    if len(parts) >= 3:
#        city_part = parts[-2]
#    elif len(parts) == 2:
#        city_part = parts[0]
#    else:
#        city_part = parts[0] if parts else None
#
#    if city_part is None:
#        return None
#
#    city_part = re.sub(r"^\d+[A-Za-z\-\s]*\s+", "", city_part).strip()
#    city_part = re.sub(r"\b\d{4,}\b", "", city_part).strip()
#    city_part = re.sub(r"[^A-Za-zÀ-ÿ'\-\s]", " ", city_part)
#    city_part = re.sub(r"\s+", " ", city_part).strip()
#
#    return city_part if city_part else None


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

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return r * c


ASPECTS = {
    "noise": [
        "quiet", "noisy", "noise", "loud", "soundproof",
        "peaceful", "calm", "sleep", "street noise"
    ],
    "location": [
        "location", "central", "center", "walk", "metro",
        "station", "transport", "far", "near", "distance"
    ],
    "cleanliness": [
        "clean", "dirty", "spotless", "hygiene", "bathroom",
        "room clean", "dust", "smell", "stain"
    ],
    "staff": [
        "staff", "service", "friendly", "helpful",
        "rude", "reception", "manager", "check in", "check-in"
    ],
    "value": [
        "value", "price", "expensive", "cheap",
        "worth", "money", "overpriced", "cost"
    ],
}

QUIET_KEYWORDS = [
    "quiet", "peaceful", "calm", "serene", "relaxing",
    "tranquil", "secluded", "sleep well", "good sleep", "not busy"
]

CROWDED_KEYWORDS = [
    "crowded", "busy", "packed", "touristy", "overcrowded",
    "noisy", "loud", "queue", "long lines", "too many people"
]


def recommend(
    df,
    city=None,
    prefer_quiet=True,
    quiet_weight=0.10,
    avoid_crowds=True,
    crowd_weight=0.10,
    prefer_cleanliness=True,
    cleanliness_weight=0.05,
    prefer_staff=True,
    staff_weight=0.05,
    prefer_location_reviews=True,
    location_review_weight=0.05,
    prefer_value=True,
    value_weight=0.05,
    distance_mode="balanced",
    distance_weight=0.05,
    top_k=10
):
    out = df.copy()

    if city is not None:
        out = out[
            out["city"].astype(str).str.strip().str.lower() == city.strip().lower()
        ].copy()

    if out.empty:
        return out

    score = out["hidden_score_base"].copy()

    if prefer_quiet:
        score += quiet_weight * out["noise_sentiment"]

    if avoid_crowds:
        score -= crowd_weight * out["crowded_negative_rate"]

    if prefer_cleanliness:
        score += cleanliness_weight * out["cleanliness_sentiment"]

    if prefer_staff:
        score += staff_weight * out["staff_sentiment"]

    if prefer_location_reviews:
        score += location_review_weight * out["location_sentiment"]

    if prefer_value:
        score += value_weight * out["value_sentiment"]

    if distance_mode == "central":
        score += distance_weight * out["centrality_score"]
    elif distance_mode == "balanced":
        score += distance_weight * out["distance_sweet_spot"]
    elif distance_mode == "outer":
        score += distance_weight * out["distance_percentile_within_city"]

    out["final_score"] = score

    return out.sort_values("final_score", ascending=False).head(top_k)


def explain(row, distance_mode="balanced"):
    reasons = []

    if row["noise_sentiment"] > 0.70:
        reasons.append("reviews are positive about quietness and noise levels")

    if row["cleanliness_sentiment"] > 0.70:
        reasons.append("guests speak positively about cleanliness")

    if row["staff_sentiment"] > 0.70:
        reasons.append("staff and service are frequently praised")

    if row["location_sentiment"] > 0.70:
        reasons.append("location is reviewed positively")

    if row["value_sentiment"] > 0.70:
        reasons.append("guests mention good value")

    if row["crowded_negative_rate"] < 0.25:
        reasons.append("fewer complaints about crowds or noise")

    if distance_mode == "central" and row["centrality_score"] > 0.75:
        reasons.append("close to the city center")

    elif distance_mode == "balanced" and row["distance_sweet_spot"] > 0.80:
        reasons.append("well positioned away from the busiest core without being too far out")

    elif distance_mode == "outer" and row["distance_percentile_within_city"] > 0.75:
        reasons.append("farther from the busiest center areas")

    if row["city_hidden_percentile"] > 0.85:
        reasons.append("stands out as an overlooked option within its city")

    if not reasons:
        reasons.append("balanced overall profile")

    return ", ".join(reasons)


@st.cache_data(show_spinner=True)
def load_and_prepare_data(path, sample_size=None):
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
            "Province_Name"
        ]
    ].copy()

    df["Positive_Review"] = df["Positive_Review"].fillna("").astype(str).str.strip()
    df["Negative_Review"] = df["Negative_Review"].fillna("").astype(str).str.strip()

    df.dropna(subset=["lat", "lng"], inplace=True)

    #df["city"] = df["Hotel_Address"].apply(extract_city).astype("string").str.strip()
    df["city"] = df["Province_Name"].astype("string").str.strip()

    for aspect, keywords in ASPECTS.items():
        df[f"{aspect}_pos_sentiment"] = df["Positive_Review"].apply(
            lambda x, kw=keywords: aspect_sentiment(x, kw)
        )

        df[f"{aspect}_neg_sentiment"] = df["Negative_Review"].apply(
            lambda x, kw=keywords: aspect_sentiment(x, kw)
        )

        df[f"{aspect}_net_sentiment"] = (
            df[f"{aspect}_pos_sentiment"] - df[f"{aspect}_neg_sentiment"]
        )

    df["quiet_in_positive"] = df["Positive_Review"].apply(
        lambda x: contains_any(x, QUIET_KEYWORDS)
    )
    df["quiet_in_negative"] = df["Negative_Review"].apply(
        lambda x: contains_any(x, QUIET_KEYWORDS)
    )
    df["crowded_in_positive"] = df["Positive_Review"].apply(
        lambda x: contains_any(x, CROWDED_KEYWORDS)
    )
    df["crowded_in_negative"] = df["Negative_Review"].apply(
        lambda x: contains_any(x, CROWDED_KEYWORDS)
    )

    df["quiet_net"] = df["quiet_in_positive"] - df["quiet_in_negative"]
    df["crowded_net"] = df["crowded_in_positive"] - df["crowded_in_negative"]

    hotel_stats = df.groupby(
        ["Hotel_Name", "city", "lat", "lng"],
        as_index=False
    ).agg(
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
        hotel_stats["city_center_lng"],
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

    hotel_stats["hidden_score_base"] = (
        0.25 * hotel_stats["avg_rating"]
        + 0.20 * hotel_stats["noise_sentiment"]
        + 0.15 * hotel_stats["cleanliness_sentiment"]
        + 0.10 * hotel_stats["staff_sentiment"]
        + 0.10 * hotel_stats["location_sentiment"]
        + 0.05 * hotel_stats["value_sentiment"]
        + 0.10 * hotel_stats["quiet_net"]
        - 0.10 * hotel_stats["crowded_negative_rate"]
        - 0.05 * hotel_stats["dataset_total_reviews"]
    )

    hotel_stats["city_hidden_percentile"] = (
        hotel_stats.groupby("city")["hidden_score_base"]
        .rank(method="average", pct=True)
    )

    return hotel_stats


st.sidebar.header("Filters & Preferences")

sample_choice = st.sidebar.selectbox(
    "Rows to process",
    options=["100,000 rows", "200,000 rows", "Full dataset"],
    index=0,
    help="Use a sample while developing. The full dataset may take longer."
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
        st.error(
            f"Could not find the dataset at `{DATA_PATH}`. "
            "Make sure `Hotel_Reviews.csv` is inside your `data/` folder."
        )
        st.stop()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

city_options = sorted(hotel_stats["city"].dropna().astype(str).unique().tolist())

if len(city_options) == 0:
    st.error("No cities were found after parsing hotel addresses.")
    st.stop()

default_city = "Paris" if "Paris" in city_options else city_options[0]

selected_city = st.sidebar.selectbox(
    "City",
    options=city_options,
    index=city_options.index(default_city)
)

st.sidebar.markdown("---")
st.sidebar.subheader("Experience preferences")

prefer_quiet = st.sidebar.checkbox("Prefer quiet stays", value=True)
quiet_weight = st.sidebar.slider(
    "Quiet preference strength",
    min_value=0.00,
    max_value=0.30,
    value=0.10,
    step=0.01,
    disabled=not prefer_quiet
)

avoid_crowds = st.sidebar.checkbox("Avoid crowd/noise complaints", value=True)
crowd_weight = st.sidebar.slider(
    "Crowd penalty strength",
    min_value=0.00,
    max_value=0.30,
    value=0.10,
    step=0.01,
    disabled=not avoid_crowds
)

prefer_cleanliness = st.sidebar.checkbox("Prioritize cleanliness", value=True)
cleanliness_weight = st.sidebar.slider(
    "Cleanliness strength",
    min_value=0.00,
    max_value=0.20,
    value=0.05,
    step=0.01,
    disabled=not prefer_cleanliness
)

prefer_staff = st.sidebar.checkbox("Prioritize staff/service", value=True)
staff_weight = st.sidebar.slider(
    "Staff/service strength",
    min_value=0.00,
    max_value=0.20,
    value=0.05,
    step=0.01,
    disabled=not prefer_staff
)

prefer_location_reviews = st.sidebar.checkbox("Prioritize positive location reviews", value=True)
location_review_weight = st.sidebar.slider(
    "Location review strength",
    min_value=0.00,
    max_value=0.20,
    value=0.05,
    step=0.01,
    disabled=not prefer_location_reviews
)

prefer_value = st.sidebar.checkbox("Prioritize value mentions", value=True)
value_weight = st.sidebar.slider(
    "Value strength",
    min_value=0.00,
    max_value=0.20,
    value=0.05,
    step=0.01,
    disabled=not prefer_value
)

st.sidebar.markdown("---")
st.sidebar.subheader("Location preference")

distance_mode_label = st.sidebar.radio(
    "Where do you want to stay?",
    options=["Central", "Balanced", "Farther from center"],
    index=1,
    help=(
        "Central favors hotels closer to the city center. "
        "Balanced favors hotels away from the busiest core without being too far. "
        "Farther from center favors hotels farther out relative to other hotels in the same city."
    )
)

distance_mode_map = {
    "Central": "central",
    "Balanced": "balanced",
    "Farther from center": "outer",
}
distance_mode = distance_mode_map[distance_mode_label]

distance_weight = st.sidebar.slider(
    "Location preference strength",
    min_value=0.00,
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

results = recommend(
    hotel_stats,
    city=selected_city,
    prefer_quiet=prefer_quiet,
    quiet_weight=quiet_weight,
    avoid_crowds=avoid_crowds,
    crowd_weight=crowd_weight,
    prefer_cleanliness=prefer_cleanliness,
    cleanliness_weight=cleanliness_weight,
    prefer_staff=prefer_staff,
    staff_weight=staff_weight,
    prefer_location_reviews=prefer_location_reviews,
    location_review_weight=location_review_weight,
    prefer_value=prefer_value,
    value_weight=value_weight,
    distance_mode=distance_mode,
    distance_weight=distance_weight,
    top_k=top_k,
)

if results.empty:
    st.warning("No recommendations found for this city.")
    st.stop()

results = results.copy()
results["explanation"] = results.apply(
    lambda row: explain(row, distance_mode=distance_mode),
    axis=1
)

st.subheader(f"Top hidden stays in {selected_city}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hotels shown", len(results))
c2.metric("Avg final score", f"{results['final_score'].mean():.3f}")
c3.metric("Avg distance", f"{results['distance_from_center_km'].mean():.2f} km")
c4.metric("Avg crowd complaints", f"{results['crowded_negative_rate'].mean():.3f}")

display_df = results[
    [
        "Hotel_Name",
        "final_score",
        "avg_rating",
        "noise_sentiment",
        "cleanliness_sentiment",
        "staff_sentiment",
        "location_sentiment",
        "value_sentiment",
        "crowded_negative_rate",
        "distance_from_center_km",
        "distance_from_center_pct",
        "city_hidden_percentile",
        "explanation",
    ]
].copy()

display_df = display_df.rename(
    columns={
        "Hotel_Name": "Hotel",
        "final_score": "Final score",
        "avg_rating": "Rating",
        "noise_sentiment": "Noise/quiet sentiment",
        "cleanliness_sentiment": "Cleanliness sentiment",
        "staff_sentiment": "Staff sentiment",
        "location_sentiment": "Location sentiment",
        "value_sentiment": "Value sentiment",
        "crowded_negative_rate": "Crowd complaint rate",
        "distance_from_center_km": "Distance from center (km)",
        "distance_from_center_pct": "Distance percentile in city",
        "city_hidden_percentile": "Hidden-gem percentile",
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
        "avg_rating": ":.3f",
        "noise_sentiment": ":.3f",
        "crowded_negative_rate": ":.3f",
        "distance_from_center_km": ":.2f",
        "distance_from_center_pct": ":.1f",
        "city_hidden_percentile": ":.3f",
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

with st.expander("Show top 5 base hidden gems from each covered city"):
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
        top_per_city[
            [
                "Hotel_Name",
                "city",
                "hidden_score_base",
                "city_hidden_percentile",
                "noise_sentiment",
                "cleanliness_sentiment",
                "staff_sentiment",
                "location_sentiment",
                "value_sentiment",
                "explanation",
            ]
        ].rename(
            columns={
                "Hotel_Name": "Hotel",
                "city": "City",
                "hidden_score_base": "Base hidden score",
                "city_hidden_percentile": "Hidden-gem percentile",
                "noise_sentiment": "Noise/quiet sentiment",
                "cleanliness_sentiment": "Cleanliness sentiment",
                "staff_sentiment": "Staff sentiment",
                "location_sentiment": "Location sentiment",
                "value_sentiment": "Value sentiment",
                "explanation": "Why it stands out",
            }
        ),
        use_container_width=True,
        hide_index=True
    )

with st.expander("Methodology"):
    st.markdown(
        """
        **Business logic summary**

        - The app identifies hidden stays within major European cities, not hidden cities.
        - Aspect-based sentiment extracts guest sentiment around noise, location, cleanliness, staff, and value.
        - Positive and negative review fields are analyzed separately.
        - Haversine distance estimates how far each hotel is from the city center.
        - Distance is converted into a within-city percentile so it is comparable within each city.
        - User controls use a checkbox + slider pattern:
            - checkbox = whether the factor matters
            - slider = how strongly that factor matters
        """
    )
