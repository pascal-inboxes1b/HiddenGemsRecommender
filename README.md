# Hidden Gems Hotels Recommender

## Overview

The **Hidden Gems Hotels Recommender** is a data science capstone project that identifies high-quality but relatively underappreciated hotels within major European cities.

Traditional hotel search results often favor highly reviewed, heavily marketed, or large chain properties. This project instead focuses on surfacing hotels that combine strong guest experience signals with lower relative popularity, helping travelers discover overlooked stays.

The project uses a content-based recommendation approach that combines:

- Review ratings
- Aspect-based sentiment analysis
- Quiet and crowded review signals
- Popularity penalties
- Geographic features
- User preference matching
- An interactive Streamlit application

---

## Business Problem

Travelers often rely on hotel rankings that are heavily influenced by review volume and overall popularity. This can make smaller or less-discovered hotels difficult to find, even when they provide strong guest experiences.

This project asks:

> Which hotels provide excellent guest experiences while remaining relatively underappreciated compared with other hotels in the same city?

The goal is to help travelers find strong hotel options that may be overlooked by traditional ranking systems.

---

## Dataset

The project uses the **515K Hotel Reviews Data in Europe** dataset from Kaggle.

Dataset source: [515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

The dataset includes:

- Hotel names
- Review scores
- Positive review text
- Negative review text
- Hotel coordinates
- Total review counts
- City-level hotel locations

Cities covered:

- Amsterdam
- Barcelona
- London
- Milan
- Paris
- Vienna

For notebook development and reporting, a **100,000-review sample** was used to support efficient local experimentation. The Streamlit application includes processing options for **100k reviews**, **200k reviews**, and the **full dataset**.

---

## Methodology

### 1. Data Preparation

The raw review-level dataset was cleaned and transformed into hotel-level features. The project uses `Metro_Area` as the city field after renaming the original `Province_Name` column.

Main preparation steps included:

- Cleaning incomplete records
- Sampling 100,000 reviews for notebook development
- Aggregating review-level records to hotel-level summaries
- Creating normalized hotel-level scoring features

---

### 2. Aspect-Based Sentiment Analysis

Instead of treating each review as one overall sentiment score, the project analyzes review text by hotel-specific aspects.

Aspects analyzed:

| Aspect | Example Meaning |
|---|---|
| Noise | quiet, noisy, loud, calm |
| Cleanliness | clean, dirty, spotless |
| Staff | helpful, friendly, rude |
| Location | central, walkable, transit access |
| Value | price, worth, expensive |

Process:

1. Split review text into sentences
2. Identify sentences related to each aspect using keyword matching
3. Score relevant sentences using TextBlob sentiment
4. Aggregate sentiment scores to the hotel level

Generated aspect features include:

- `noise_sentiment`
- `cleanliness_sentiment`
- `staff_sentiment`
- `location_sentiment`
- `value_sentiment`

---

### 3. Quiet and Crowded NLP Features

Additional NLP features were engineered to capture whether hotels are described as quiet, calm, busy, noisy, or crowded.

Key features:

- `quiet_net`
- `crowded_negative_rate`

These features help distinguish hotels that may be a better match for travelers seeking calm stays versus those who prefer livelier environments.

---

### 4. Geographic Features

Hotel coordinates were used to calculate distance from city center using the Haversine formula.

Generated geographic features:

- `distance_from_center_km`
- `distance_percentile_within_city`
- `distance_sweet_spot`
- `centrality_score`

These features allow the recommender to support different location preferences:

- Central
- Balanced
- Farther from center

---

### 5. Hidden Gem Base Scoring

A baseline hidden-gem score was created using expert-defined weights. The base score rewards strong guest experience signals while applying a small penalty for higher review volume.

Base scoring weights:

```python
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
```

Popularity is represented using:

```python
hotel_stats["log_review_count"] = np.log1p(
    hotel_stats["dataset_total_reviews"]
)
```

The output of this stage is:

- `hidden_score_base`

---

### 6. Personalized Recommendation Engine

The final recommender separates baseline hotel quality from user preference matching.

User controls include:

- Quiet/calm preference
- Crowd/noise avoidance preference
- Location preference
- Preference strength sliders

The final recommender outputs:

- `quiet_match_score`
- `calm_quality_score`
- `crowd_match_score`
- `distance_match_score`
- `final_score`
- `city_final_percentile`

This allows two users to receive different hotel rankings from the same hotel inventory.

---

## Streamlit Application

The project includes an interactive Streamlit app that allows users to:

- Select a city
- Choose quiet/calm preferences
- Choose crowd/noise preferences
- Choose location preferences
- Adjust preference strength
- View ranked hotel recommendations
- Explore recommendations on an interactive map
- Review explainability metrics for each recommendation

The app is located in:

```text
app/app.py
```

Run the app from the project root using:

```bash
streamlit run app/app.py
```

---

## Key Results

The final report includes five project figures:

1. **Distribution of Hidden Gem Scores**  
   Shows the spread of baseline hidden-gem scores across hotels.

2. **Popularity vs Hidden Gem Score**  
   Shows that review popularity alone does not determine hidden-gem quality.

3. **Top Hidden Gems by City**  
   Compares average top hidden-gem scores across the six supported cities.

4. **Impact of User Preferences on Recommendations**  
   Demonstrates that personalized preference settings materially change recommendation rankings. In the Paris example, 7 of the top 10 recommendations changed when traveler preferences changed.

5. **Streamlit Application Interface**  
   Demonstrates the completed end-to-end interactive recommender application.

---

## Recommendations

This project supports three practical recommendations:

1. **Use hidden-gem scoring to complement popularity-based hotel rankings.**  
   Hotels with strong guest experience signals but lower relative visibility can be surfaced more effectively.

2. **Allow travelers to personalize recommendation logic.**  
   Quiet, crowd, and location preferences meaningfully affect which hotels are most relevant.

3. **Use explainability features to build user trust.**  
   Intermediate scores such as calm match, crowd match, and distance match help explain why a hotel was recommended.

---

## Limitations

Current limitations include:

- Sentiment analysis uses TextBlob and keyword-based aspect matching, which may miss complex language patterns.
- The scoring system uses expert-defined weights rather than learned weights from user behavior.
- The notebook report figures are based on a 100,000-review development sample.
- Hotel availability, pricing, amenities, and booking platform data are not included.
- The system is content-based and does not use collaborative filtering.

---

## Future Work

Potential future improvements include:

- Transformer-based sentiment analysis
- Topic modeling or BERTopic review clustering
- Incorporating hotel price and availability data
- Adding public transit and attraction-distance features
- Learning weights from user clicks, saves, or bookings
- Adding collaborative filtering if user interaction data becomes available
- Improving deployment and caching for full-dataset processing

---

## Project Structure

```text
HiddenGemsRecommender/
├── README.md
├── requirements.txt
├── app/
│   └── app.py
├── notebooks/
│   └── Hidden_Gems_Final_Notebook.ipynb
├── reports/
│   ├── Hidden_Gems_Hotels_Recommender_Final_Report.pdf
│   ├── Hidden_Gems_Hotels_Recommender_Model_Metrics.txt
│   └── figures/
│       ├── Figure1_HiddenScoreDistribution.png
│       ├── Figure2_PopularityVsHiddenScore.png
│       ├── Figure3_TopHiddenGemsByCity.png
│       ├── Figure4_UserPreferenceImpact.png
│       └── Figure5_StreamlitAppScreenshot.png
└── presentation/
    └── Hidden_Gems_Hotels_Recommender_Presentation.pptx
```

The final report, model metrics file, and report figures are stored in the `reports/` folder. The Streamlit application is stored in `app/`, and the final notebook is stored in `notebooks/`.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TextBlob
- Plotly
- Streamlit
- Matplotlib
- Jupyter Notebook

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```

---

## Deliverables

Final deliverables include:

- Final project report PDF
- Model metrics file
- Cleaned README
- Final notebook
- Streamlit application
- Report figures
- Presentation slides

---

## Author

James F  
Data Science Capstone Project  
2026
