import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Brewlytics | Indian Café Intelligence",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual theme
st.markdown("""
<style>
:root { --ink:#1d2733; --muted:#657386; --cream:#fffaf2; --coffee:#6b3f2a; --mint:#d8f3e7; }
.stApp { background: linear-gradient(180deg, #fffaf2 0%, #f7f8fb 45%, #f7f8fb 100%); }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
.hero { padding: 1.5rem 1.7rem; border-radius: 22px; color: white;
        background: linear-gradient(120deg, #2d4739 0%, #6b3f2a 62%, #bd7b46 100%);
        box-shadow: 0 12px 30px rgba(40,45,40,.12); margin-bottom: 1.1rem; }
.hero h1 { color:white; margin:0; font-size:2.25rem; letter-spacing:-.04em; }
.hero p { color:#f4e9da; margin:.45rem 0 0 0; font-size:1rem; }
.kpi { background: rgba(255,255,255,.92); border:1px solid #e9e2d8; border-radius:18px;
       padding: 1rem 1.1rem; min-height:118px; box-shadow:0 5px 16px rgba(35,40,35,.04); }
.kpi-label { color:#657386; font-size:.82rem; font-weight:600; }
.kpi-value { color:#24352c; font-size:1.8rem; font-weight:800; margin-top:.3rem; }
.kpi-note { color:#7a8795; font-size:.75rem; margin-top:.2rem; }
.section-head { color:#26372e; font-weight:800; font-size:1.25rem; padding:.2rem 0 .6rem 0; }
.small-note { color:#657386; font-size:.85rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e9e2d8; border-radius:16px; padding:14px 16px; }
[data-testid="stSidebar"] { background:#f1eee7; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "cafe_reviews_cleaned.csv"

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        d = pd.read_csv(DATA_PATH)
    else:
        d = pd.read_csv("data/cafe_reviews_cleaned.csv")
    for c in ["Name", "Cuisine", "City", "Review", "Price_Tier", "Rating_Band", "Sentiment_Label"]:
        if c in d.columns:
            d[c] = d[c].fillna("").astype(str)
    d["Overall_Rating"] = pd.to_numeric(d["Overall_Rating"], errors="coerce")
    d["Rate for two"] = pd.to_numeric(d["Rate for two"], errors="coerce")
    d["Review_Length"] = pd.to_numeric(d["Review_Length"], errors="coerce")
    d["Cuisine_Count"] = pd.to_numeric(d["Cuisine_Count"], errors="coerce")
    d["Review_Opportunity_Score"] = pd.to_numeric(d["Review_Opportunity_Score"], errors="coerce")
    return d

df = load_data()

def money(v):
    return "₹{:,.0f}".format(v) if pd.notna(v) else "—"

def cuisine_tokens(series):
    vals = set()
    for item in series.dropna().astype(str):
        for token in item.split(","):
            token = token.strip()
            if token:
                vals.add(token)
    return sorted(vals)

def build_model(data):
    # Weak labels are derived from existing ratings; this is not human-annotated sentiment.
    train = data[data["Overall_Rating"].notna() & data["Review"].str.len().gt(8)].copy()
    train["Target"] = train["Overall_Rating"].apply(
        lambda r: "Positive" if r >= 4.0 else ("Negative" if r <= 2.5 else "Neutral")
    )
    if train["Target"].nunique() < 2 or len(train) < 20:
        return None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(
        train["Review"], train["Target"], test_size=0.25, random_state=42, stratify=train["Target"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, accuracy_score(y_test, pred), y_test, pred

@st.cache_resource
def get_model():
    return build_model(df)

model, model_accuracy, y_test, y_pred = get_model()

# Sidebar filters
st.sidebar.markdown("## 🎛️ Explore the data")
st.sidebar.caption("Filters update the dashboard charts and café table.")
cities = sorted([x for x in df["City"].dropna().unique() if x])
selected_cities = st.sidebar.multiselect("City", cities, default=cities)
price_tiers = ["Budget (≤₹400)", "Mid-range (₹401–₹700)", "Premium (₹701–₹1,000)", "Luxury (>₹1,000)", "Unknown"]
selected_tiers = st.sidebar.multiselect("Price tier (listed cost for two)", price_tiers, default=price_tiers)
rating_min = st.sidebar.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1)
rating_max = st.sidebar.slider("Maximum rating", 0.0, 5.0, 5.0, 0.1)
sentiment_opts = ["Positive", "Neutral", "Negative", "Unlabelled"]
selected_sentiments = st.sidebar.multiselect("Rating-derived sentiment", sentiment_opts, default=sentiment_opts)
search_text = st.sidebar.text_input("Search café, cuisine or review", "")
cuisine_opts = cuisine_tokens(df["Cuisine"])
selected_cuisines = st.sidebar.multiselect("Cuisine contains", cuisine_opts)
min_review_length = st.sidebar.slider("Minimum review length", 0, int(max(10, df["Review_Length"].max())), 0, 10)
st.sidebar.markdown("---")
st.sidebar.caption("Source file: reviews.csv • Ratings and listed price are source fields. No dates, transactions, costs or profits are present.")

filtered = df.copy()
filtered = filtered[filtered["City"].isin(selected_cities)]
filtered = filtered[filtered["Price_Tier"].isin(selected_tiers)]
filtered = filtered[
    filtered["Overall_Rating"].isna()
    | ((filtered["Overall_Rating"] >= rating_min) & (filtered["Overall_Rating"] <= rating_max))
]
filtered = filtered[filtered["Sentiment_Label"].isin(selected_sentiments)]
filtered = filtered[filtered["Review_Length"].fillna(0) >= min_review_length]
if selected_cuisines:
    pat = "|".join(re.escape(x) for x in selected_cuisines)
    filtered = filtered[filtered["Cuisine"].str.contains(pat, case=False, na=False)]
if search_text.strip():
    q = re.escape(search_text.strip())
    mask = (
        filtered["Name"].str.contains(q, case=False, na=False)
        | filtered["Cuisine"].str.contains(q, case=False, na=False)
        | filtered["Review"].str.contains(q, case=False, na=False)
    )
    filtered = filtered[mask]

# Header
st.markdown("""
<div class="hero">
  <h1>☕ Brewlytics</h1>
  <p>Indian Café Review Intelligence · customer voice, affordability & business discovery</p>
</div>
""", unsafe_allow_html=True)
st.markdown("This dashboard explores café reviews and listed prices. It is a customer-insight prototype, **not an accounting system**: this dataset does not include transactions, dated sales, operating expenses or actual profit.")

n_reviews = len(filtered)
n_cafes = filtered["Name"].nunique() if n_reviews else 0
avg_rating = filtered["Overall_Rating"].mean() if n_reviews else np.nan
median_price = filtered["Rate for two"].median() if n_reviews else np.nan
positive_share = (filtered["Sentiment_Label"].eq("Positive").mean() * 100) if n_reviews else np.nan

cols = st.columns(4)
kpis = [
    ("Reviews in view", f"{n_reviews:,}", "Rows after current filters"),
    ("Cafés in view", f"{n_cafes:,}", "Distinct café names"),
    ("Average listed rating", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "—", "Mean of available ratings"),
    ("Median listed cost for two", money(median_price), "Not actual average bill"),
]
for col, (label, value, note) in zip(cols, kpis):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

st.markdown("")
tab_overview, tab_customer, tab_cafes, tab_ml, tab_data = st.tabs([
    "✨ Overview", "💬 Customer Voice", "🏙️ Café Explorer", "🧠 ML Lab", "🧾 Data & Export"
])

with tab_overview:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-head">Rating distribution</div>', unsafe_allow_html=True)
        rating_data = filtered.dropna(subset=["Overall_Rating"])
        if len(rating_data):
            fig = px.histogram(rating_data, x="Overall_Rating", nbins=18, color_discrete_sequence=["#6b8f71"])
            fig.update_layout(template="plotly_white", height=330, xaxis_title="Listed rating", yaxis_title="Review records", bargap=.08, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rated records match the selected filters.")
    with right:
        st.markdown('<div class="section-head">Listed price tiers</div>', unsafe_allow_html=True)
        price_data = filtered["Price_Tier"].value_counts().rename_axis("Price tier").reset_index(name="Reviews")
        if len(price_data):
            fig = px.pie(price_data, names="Price tier", values="Reviews", hole=.58, color_discrete_sequence=["#2d4739","#6b8f71","#d6a66b","#a85f45","#b8b8b8"])
            fig.update_layout(template="plotly_white", height=330, margin=dict(l=10,r=10,t=10,b=10), legend_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No records match the selected filters.")
    st.markdown('<div class="section-head">City comparison</div>', unsafe_allow_html=True)
    city_stats = filtered.groupby("City", dropna=False).agg(
        Reviews=("Review", "count"),
        Cafes=("Name", "nunique"),
        Avg_Rating=("Overall_Rating", "mean"),
        Median_Price_For_Two=("Rate for two", "median")
    ).reset_index().sort_values("Reviews", ascending=False)
    if len(city_stats):
        fig = px.bar(city_stats, x="City", y="Reviews", color="Avg_Rating", hover_data=["Cafes", "Median_Price_For_Two"], color_continuous_scale=["#f0d6b8","#2d4739"])
        fig.update_layout(template="plotly_white", height=380, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="Review records")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records match the selected filters.")

with tab_customer:
    st.markdown('<div class="section-head">What customers are saying</div>', unsafe_allow_html=True)
    st.caption("Sentiment is a proxy label derived from the existing rating: ≥4 positive, ≤2.5 negative, otherwise neutral. It is not an independently annotated emotion label.")
    left, right = st.columns([1, 1])
    with left:
        sent = filtered["Sentiment_Label"].value_counts().rename_axis("Rating-derived label").reset_index(name="Reviews")
        if len(sent):
            fig = px.bar(sent, x="Rating-derived label", y="Reviews", color="Rating-derived label",
                         color_discrete_map={"Positive":"#2d7a55","Neutral":"#d6a66b","Negative":"#b84a4a","Unlabelled":"#9aa3ad"})
            fig.update_layout(template="plotly_white", height=320, showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Review examples**")
        sample = filtered[["Name", "City", "Overall_Rating", "Review", "Sentiment_Label"]].dropna(subset=["Review"]).head(5)
        if len(sample):
            for _, r in sample.iterrows():
                rating = f"{r['Overall_Rating']:.1f}/5" if pd.notna(r["Overall_Rating"]) else "Unrated"
                st.markdown(f"**{r['Name']} · {r['City']} · {rating}**")
                st.write(r["Review"])
                st.caption(f"Rating-derived label: {r['Sentiment_Label']}")
                st.divider()
        else:
            st.info("No reviews match the filters.")
    st.markdown("**Most common cuisine tags**")
    tag_counts = {}
    for value in filtered["Cuisine"].dropna().astype(str):
        for tag in value.split(","):
            tag = tag.strip()
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if tag_counts:
        tags = pd.DataFrame(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:15], columns=["Cuisine tag", "Review records"])
        fig = px.bar(tags.sort_values("Review records"), x="Review records", y="Cuisine tag", orientation="h", color_discrete_sequence=["#bd7b46"])
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_cafes:
    st.markdown('<div class="section-head">Café comparison explorer</div>', unsafe_allow_html=True)
    cafe_stats = filtered.groupby(["Name", "City"], dropna=False).agg(
        Review_Records=("Review", "count"),
        Listed_Rating=("Overall_Rating", "mean"),
        Listed_Price_For_Two=("Rate for two", "median"),
        Common_Cuisine=("Cuisine", lambda x: x.mode().iloc[0] if not x.mode().empty else "")
    ).reset_index()
    cafe_stats["Listed_Rating"] = cafe_stats["Listed_Rating"].round(2)
    cafe_stats = cafe_stats.sort_values(["Listed_Rating", "Review_Records"], ascending=[False, False], na_position="last")
    show_all = st.checkbox("Show all matching cafés", value=False)
    st.dataframe(cafe_stats if show_all else cafe_stats.head(30), use_container_width=True, hide_index=True)
    if len(cafe_stats):
        st.download_button("⬇️ Download café comparison CSV", cafe_stats.to_csv(index=False).encode("utf-8"), "cafe_comparison.csv", "text/csv")
    st.markdown("**Review-opportunity score**")
    st.caption("A descriptive heuristic based on listed rating (70%) and relative affordability (30%). It does not measure revenue, demand, market share or profitability.")
    score_view = filtered.groupby(["Name", "City"], dropna=False).agg(
        Listed_Rating=("Overall_Rating", "mean"),
        Listed_Price_For_Two=("Rate for two", "median"),
        Reviews=("Review", "count"),
        Review_Opportunity_Score=("Review_Opportunity_Score", "mean")
    ).reset_index().sort_values("Review_Opportunity_Score", ascending=False).head(15)
    if len(score_view):
        fig = px.scatter(score_view, x="Listed_Price_For_Two", y="Listed_Rating", size="Reviews", color="City",
                         hover_name="Name", hover_data=["Review_Opportunity_Score"], size_max=28)
        fig.update_layout(template="plotly_white", height=440, xaxis_title="Listed cost for two (₹)", yaxis_title="Average listed rating", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_ml:
    st.markdown('<div class="section-head">🧠 Review sentiment model</div>', unsafe_allow_html=True)
    st.write("A TF-IDF + Logistic Regression model learns text patterns from reviews. Training labels are derived from star ratings, so this is **weak supervision** rather than human-verified sentiment.")
    if model is not None:
        a, b, c = st.columns(3)
        a.metric("Model", "TF-IDF + Logistic Regression")
        b.metric("Holdout accuracy", f"{model_accuracy:.1%}" if model_accuracy is not None else "—")
        c.metric("Training records", f"{df['Overall_Rating'].notna().sum():,}")
        st.caption("Accuracy is measured against rating-derived labels on a random holdout split. It should not be interpreted as proof of true human sentiment accuracy.")
        st.markdown("**Try a review**")
        review_input = st.text_area("Paste a customer review", placeholder="Example: Great coffee, friendly staff and a comfortable place to work...", height=120)
        if st.button("Predict rating-derived sentiment", type="primary"):
            if review_input.strip():
                pred = model.predict([review_input])[0]
                probs = model.predict_proba([review_input])[0]
                classes = model.named_steps["classifier"].classes_
                confidence = float(probs[list(classes).index(pred)])
                emoji = {"Positive":"😊", "Neutral":"😐", "Negative":"🙁"}.get(pred, "💬")
                st.success(f"{emoji} Predicted label: **{pred}** · model probability {confidence:.0%}")
                st.caption("This prediction is experimental and may be wrong, especially for sarcasm, mixed-language reviews or short text.")
            else:
                st.warning("Enter a review first.")
        if y_test is not None and y_pred is not None:
            with st.expander("View holdout classification report"):
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
                labels = sorted(pd.Series(y_test).unique())
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                fig = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto", color_continuous_scale="Greens",
                                labels=dict(x="Predicted", y="Actual", color="Records"))
                fig.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough rated reviews to train the model. Check that the dataset contains usable rating values.")

with tab_data:
    st.markdown('<div class="section-head">Data quality & exports</div>', unsafe_allow_html=True)
    st.write("Derived columns added for this project:")
    st.markdown("""
    - **Price_Tier:** groups the listed cost for two into budget, mid-range, premium and luxury bands.
    - **Rating_Band:** groups available ratings into low, average, good and excellent.
    - **Sentiment_Label:** a transparent label derived from rating thresholds, not human-labelled sentiment.
    - **Review_Length / Cuisine_Count:** simple text and cuisine features.
    - **Review_Opportunity_Score:** a heuristic for exploration, not a financial metric.
    """)
    quality = pd.DataFrame({
        "Field": df.columns,
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered review data", filtered.to_csv(index=False).encode("utf-8"), "filtered_cafe_reviews.csv", "text/csv")
    st.download_button("⬇️ Download cleaned full dataset", df.to_csv(index=False).encode("utf-8"), "cafe_reviews_cleaned.csv", "text/csv")
    st.markdown("**Limitations to keep visible in a professional portfolio**")
    st.warning("The source file has no date, customer ID, transaction quantity, actual sales, ingredient cost, rent, salaries or operating expenses. Do not claim this dataset calculates real daily sales, unique customers, profit or 2022–2026 trends. Add real POS and expense data to support those features.")

st.markdown("---")
st.markdown('<p class="small-note">Brewlytics · Portfolio prototype built with Streamlit, Pandas, Plotly and Scikit-learn · Source data supplied by project owner.</p>', unsafe_allow_html=True)
