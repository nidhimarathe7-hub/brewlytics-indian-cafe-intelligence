from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

st.set_page_config(page_title="Brewlytics | Café Intelligence", page_icon="☕", layout="wide", initial_sidebar_state="expanded")

# Cohesive cream / forest / amber visual system
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#1d3028;--muted:#66756c;--cream:#fbf8f1;--forest:#214e3b;--leaf:#4f8065;--gold:#d6a85f;--line:#e7e3d8;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:linear-gradient(160deg,#fbf8f1 0%,#f6f8f5 60%,#f3f6f2 100%);color:var(--ink)}
.block-container{padding-top:1.35rem;padding-bottom:2.5rem;max-width:1500px}
[data-testid="stSidebar"]{background:#f0eee5;border-right:1px solid #e1dfd4}
.hero{padding:1.6rem 1.8rem;border-radius:24px;color:#fff;background:linear-gradient(115deg,#173d2d 0%,#2c6448 58%,#b58a4a 140%);box-shadow:0 12px 28px rgba(31,66,47,.14);margin-bottom:1.05rem;position:relative;overflow:hidden}
.hero h1{font-family:'Manrope',sans-serif;color:#fff;font-size:2.45rem;letter-spacing:-.055em;margin:0;font-weight:800}
.hero p{color:#e9f1e9;margin:.45rem 0 0;font-size:1rem}
.hero .pill{display:inline-block;margin-top:.9rem;border:1px solid rgba(255,255,255,.32);background:rgba(255,255,255,.1);padding:.32rem .7rem;border-radius:999px;font-size:.76rem;color:#f7f4e9}
.section-title{font-family:'Manrope',sans-serif;color:#214e3b;font-size:1.22rem;font-weight:800;margin:.2rem 0 .5rem}
.kpi{background:rgba(255,255,255,.88);border:1px solid #e7e3d8;border-radius:18px;padding:1rem 1.1rem;min-height:116px;box-shadow:0 5px 16px rgba(35,55,40,.035)}
.kpi-label{color:#68766d;font-size:.78rem;font-weight:700;letter-spacing:.02em}.kpi-value{font-family:'Manrope',sans-serif;color:#214e3b;font-size:1.8rem;font-weight:800;margin-top:.35rem}.kpi-note{color:#7a867e;font-size:.73rem;margin-top:.2rem}
.insight{background:#edf4ed;border:1px solid #d8e6d9;border-left:4px solid #4f8065;border-radius:12px;padding:.85rem 1rem;margin:.35rem 0;color:#244533}
.warning-note{background:#fff5e3;border:1px solid #f0dfba;border-radius:12px;padding:.8rem 1rem;color:#624a1d}
div[data-testid="stMetric"]{background:white;border:1px solid var(--line);border-radius:14px;padding:12px 15px}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:#eeefe8;border-radius:14px;padding:6px}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:9px 14px;font-weight:700}
.stTabs [aria-selected="true"]{background:#fff!important;color:#214e3b!important}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:700;border:1px solid #d5dfd5}
.small{font-size:.83rem;color:#6a786e}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    candidates = [BASE_DIR / "data" / "cafe_reviews_cleaned.csv", BASE_DIR / "cafe_reviews_cleaned.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Dataset missing. Upload cafe_reviews_cleaned.csv to the repository root or a data/ folder.")
    data = pd.read_csv(csv_path)
    required = ["Name", "Overall_Rating", "Cuisine", "Rate for two", "City", "Review"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    for col in ["Name", "Cuisine", "City", "Review"]:
        data[col] = data[col].fillna("").astype(str)
    data["Overall_Rating"] = pd.to_numeric(data["Overall_Rating"], errors="coerce")
    data["Rate for two"] = pd.to_numeric(data["Rate for two"], errors="coerce")
    data["Review_Length"] = data["Review"].str.len()
    if "Price_Tier" not in data:
        data["Price_Tier"] = pd.cut(data["Rate for two"], [-np.inf,400,700,1000,np.inf], labels=["Budget (≤₹400)","Mid-range (₹401–₹700)","Premium (₹701–₹1,000)","Luxury (>₹1,000)"]).astype("object")
    data["Price_Tier"] = data["Price_Tier"].fillna("Unknown")
    data["Sentiment_Label"] = data["Overall_Rating"].apply(lambda r: "Positive" if pd.notna(r) and r >= 4 else ("Negative" if pd.notna(r) and r <= 2.5 else ("Neutral" if pd.notna(r) else "Unlabelled")))
    data["Cuisine_Count"] = data["Cuisine"].apply(lambda x: len([t for t in x.split(",") if t.strip()]))
    return data

@st.cache_resource
def train_model(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=.25, random_state=42, stratify=labels)
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=12000, stop_words="english")), ("classifier", LogisticRegression(max_iter=1600, class_weight="balanced", random_state=42))])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    return pipe, y_test, predictions, accuracy_score(y_test, predictions)

def money(value):
    return f"₹{value:,.0f}" if pd.notna(value) else "—"

def cuisine_tokens(series):
    return sorted({part.strip() for value in series.dropna().astype(str) for part in value.split(",") if part.strip()})

def chart_style(fig, height=350):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=12,r=12,t=28,b=12), font=dict(family="DM Sans, sans-serif", color="#31473a"), legend_title_text="")
    fig.update_xaxes(showgrid=False, linecolor="#e5e7df")
    fig.update_yaxes(gridcolor="#edf0e9", zeroline=False)
    return fig

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info("Check the GitHub repository and ensure app.py and the CSV are committed. The app supports both /data/cafe_reviews_cleaned.csv and a CSV at the repository root.")
    st.stop()

# Sidebar: filters are applied together and affect all analysis sections.
st.sidebar.markdown("# 🎛️ Market filters")
st.sidebar.caption("Every selection updates the dashboard, tables and downloads.")
all_cities = sorted(x for x in df["City"].unique() if x and x.lower() != "nan")
all_tiers = sorted(df["Price_Tier"].dropna().astype(str).unique())
all_sentiments = ["Positive", "Neutral", "Negative", "Unlabelled"]
city_sel = st.sidebar.multiselect("City / market", all_cities, default=all_cities)
tier_sel = st.sidebar.multiselect("Listed price tier", all_tiers, default=all_tiers)
sent_sel = st.sidebar.multiselect("Rating-derived label", all_sentiments, default=all_sentiments)
rating_values = df["Overall_Rating"].dropna()
rmin = float(rating_values.min()) if not rating_values.empty else 0.0
rmax = float(rating_values.max()) if not rating_values.empty else 5.0
rating_range = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0, 5.0), 0.1)
cuisine_sel = st.sidebar.multiselect("Cuisine tags", cuisine_tokens(df["Cuisine"]))
min_price = int(max(0, df["Rate for two"].min())) if df["Rate for two"].notna().any() else 0
max_price = int(df["Rate for two"].max()) if df["Rate for two"].notna().any() else 5000
price_range = st.sidebar.slider("Listed cost for two (₹)", min_price, max(max_price, min_price+1), (min_price, max(max_price, min_price+1)))
max_len = int(max(10, df["Review_Length"].max()))
review_length = st.sidebar.slider("Minimum review length", 0, max_len, 0, 10)
search = st.sidebar.text_input("Search café, city, cuisine or review", placeholder="e.g. coffee, service, Mumbai")
include_unrated = st.sidebar.checkbox("Include records without ratings", value=True)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} review records · {df['Name'].nunique():,} café names · {df['City'].nunique():,} cities")

f = df[df["City"].isin(city_sel) & df["Price_Tier"].isin(tier_sel) & df["Sentiment_Label"].isin(sent_sel)].copy()
f = f[f["Rate for two"].between(price_range[0], price_range[1], inclusive="both") | f["Rate for two"].isna()]
f = f[(f["Overall_Rating"].between(rating_range[0], rating_range[1], inclusive="both")) | (include_unrated & f["Overall_Rating"].isna())]
f = f[f["Review_Length"] >= review_length]
if cuisine_sel:
    pattern = "|".join(re.escape(x) for x in cuisine_sel)
    f = f[f["Cuisine"].str.contains(pattern, case=False, na=False)]
if search.strip():
    q = re.escape(search.strip())
    mask = False
    for col in ["Name", "City", "Cuisine", "Review"]:
        mask = mask | f[col].str.contains(q, case=False, na=False)
    f = f[mask]

st.markdown("""
<div class="hero"><h1>☕ Brewlytics</h1><p>Indian Café Review Intelligence · discover customer patterns, compare markets, make evidence-led decisions.</p><span class="pill">BUSINESS INTELLIGENCE</span> <span class="pill">MACHINE LEARNING</span> <span class="pill">INTERACTIVE ANALYTICS</span></div>
""", unsafe_allow_html=True)
st.markdown('<div class="warning-note">Scope note: this dataset contains café listings, ratings, cuisine tags, listed cost for two and review text. It does not contain transaction dates, actual sales, operating costs or profit. This dashboard does not invent financial results.</div>', unsafe_allow_html=True)

n_reviews = len(f); n_cafes = f["Name"].nunique() if n_reviews else 0
avg_rating = f["Overall_Rating"].mean() if n_reviews else np.nan
median_price = f["Rate for two"].median() if n_reviews else np.nan
positive_share = (f["Sentiment_Label"].eq("Positive").mean()*100) if n_reviews else np.nan
kpis = [("REVIEWS IN VIEW", f"{n_reviews:,}", "After all active filters"), ("CAFÉS IN VIEW", f"{n_cafes:,}", "Distinct café names"), ("AVERAGE RATING", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "—", "Mean available rating"), ("MEDIAN LISTED PRICE", money(median_price), "Listed cost for two"), ("POSITIVE-LABEL SHARE", f"{positive_share:.1f}%" if pd.notna(positive_share) else "—", "Rating-derived, not human-coded")]
cols = st.columns(5)
for col, (label, value, note) in zip(cols, kpis):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

st.write("")
tabs = st.tabs(["✦ Executive overview", "♡ Customer voice", "⌕ Café explorer", "◈ ML lab", "▤ Data & export"])

with tabs[0]:
    st.markdown('<div class="section-title">Market snapshot</div>', unsafe_allow_html=True)
    left, right = st.columns([1.08, 1])
    with left:
        rd = f.dropna(subset=["Overall_Rating"])
        if len(rd):
            fig = px.histogram(rd, x="Overall_Rating", nbins=18, color_discrete_sequence=["#4f8065"], labels={"Overall_Rating":"Rating out of 5", "count":"Review records"})
            st.plotly_chart(chart_style(fig, 320), use_container_width=True)
        else: st.info("No rated reviews match these filters.")
    with right:
        tier_counts = f["Price_Tier"].value_counts().rename_axis("Price tier").reset_index(name="Review records")
        if len(tier_counts):
            fig = px.pie(tier_counts, names="Price tier", values="Review records", hole=.62, color_discrete_sequence=["#214e3b","#4f8065","#d6a85f","#bc7957","#aeb9af"])
            fig.update_traces(textposition="inside", textinfo="percent")
            st.plotly_chart(chart_style(fig, 320), use_container_width=True)
        else: st.info("No records match these filters.")
    st.markdown('<div class="section-title">Compare city markets</div>', unsafe_allow_html=True)
    city = f.groupby("City", as_index=False).agg(Reviews=("Review","size"), Cafés=("Name","nunique"), Average_Rating=("Overall_Rating","mean"), Median_Cost_For_Two=("Rate for two","median")).sort_values("Reviews", ascending=False)
    if len(city):
        fig = px.bar(city, x="City", y="Reviews", color="Average_Rating", hover_data=["Cafés", "Median_Cost_For_Two"], color_continuous_scale=["#e6d4ad","#214e3b"], labels={"Reviews":"Review records", "Average_Rating":"Avg rating"})
        st.plotly_chart(chart_style(fig, 370), use_container_width=True)
        st.dataframe(city.round(2), use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title">Evidence-led quick insights</div>', unsafe_allow_html=True)
    if len(f):
        most_reviewed = f.groupby("City").size().sort_values(ascending=False)
        top_tier = f["Price_Tier"].value_counts()
        negative = f[f["Sentiment_Label"] == "Negative"]
        st.markdown(f'<div class="insight"><b>Largest visible market:</b> {most_reviewed.index[0]} has the most review records in the current selection ({most_reviewed.iloc[0]:,}). This is dataset coverage, not market share.</div>', unsafe_allow_html=True)
        if len(top_tier): st.markdown(f'<div class="insight"><b>Most common price tier:</b> {top_tier.index[0]} ({top_tier.iloc[0]:,} review records in the selected view).</div>', unsafe_allow_html=True)
        if len(negative):
            st.markdown(f'<div class="insight"><b>Customer-experience investigation:</b> {len(negative):,} records have ratings ≤2.5. Review these comments to identify possible service or product issues.</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="insight"><b>Customer-experience investigation:</b> No low-rating records appear in the current filtered view. Broaden filters to check more feedback.</div>', unsafe_allow_html=True)
    else: st.info("Broaden your filters to see insights.")

with tabs[1]:
    st.markdown('<div class="section-title">Customer voice & review patterns</div>', unsafe_allow_html=True)
    st.caption("Sentiment classes are derived from star ratings: positive ≥4.0, negative ≤2.5, otherwise neutral. They are proxies, not human-annotated emotion labels.")
    a,b = st.columns(2)
    with a:
        sc = f["Sentiment_Label"].value_counts().rename_axis("Rating-derived class").reset_index(name="Records")
        if len(sc):
            fig = px.bar(sc, x="Rating-derived class", y="Records", color="Rating-derived class", color_discrete_map={"Positive":"#4f8065","Neutral":"#d6a85f","Negative":"#b85f4a","Unlabelled":"#aeb9af"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_style(fig, 310), use_container_width=True)
    with b:
        cuisine_count = {}
        for value in f["Cuisine"].astype(str):
            for tag in value.split(","):
                tag = tag.strip()
                if tag: cuisine_count[tag] = cuisine_count.get(tag,0)+1
        if cuisine_count:
            cdf = pd.DataFrame(sorted(cuisine_count.items(), key=lambda x:x[1], reverse=True)[:12], columns=["Cuisine tag","Records"]).sort_values("Records")
            fig = px.bar(cdf, x="Records", y="Cuisine tag", orientation="h", color_discrete_sequence=["#b58a4a"])
            st.plotly_chart(chart_style(fig, 310), use_container_width=True)
    st.markdown('<div class="section-title">Searchable review examples</div>', unsafe_allow_html=True)
    show_reviews = f[["Name","City","Overall_Rating","Price_Tier","Review","Sentiment_Label"]].copy()
    show_reviews["Overall_Rating"] = show_reviews["Overall_Rating"].round(1)
    st.dataframe(show_reviews.head(100), use_container_width=True, hide_index=True, column_config={"Review":st.column_config.TextColumn("Customer review", width="large"), "Overall_Rating":st.column_config.NumberColumn("Rating", format="%.1f / 5")})
    st.caption("Showing up to 100 records for readability; use the sidebar search and filters to narrow the review set.")

with tabs[2]:
    st.markdown('<div class="section-title">Café comparison explorer</div>', unsafe_allow_html=True)
    cafe = f.groupby(["Name","City"], as_index=False).agg(Review_Records=("Review","size"), Average_Rating=("Overall_Rating","mean"), Listed_Cost_For_Two=("Rate for two","median"), Cuisine_Tags=("Cuisine",lambda x: x.mode().iloc[0] if not x.mode().empty else ""))
    cafe["Average_Rating"] = cafe["Average_Rating"].round(2)
    min_reviews = st.slider("Minimum review records for comparison", 1, max(1, int(cafe["Review_Records"].max()) if len(cafe) else 1), 1)
    cafe = cafe[cafe["Review_Records"] >= min_reviews]
    sort_by = st.selectbox("Rank cafés by", ["Average_Rating","Review_Records","Listed_Cost_For_Two"], format_func=lambda x:{"Average_Rating":"Highest average rating","Review_Records":"Most review records","Listed_Cost_For_Two":"Highest listed cost for two"}[x])
    cafe = cafe.sort_values(sort_by, ascending=(sort_by != "Listed_Cost_For_Two"), na_position="last")
    st.dataframe(cafe.head(150), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download café comparison CSV", cafe.to_csv(index=False).encode("utf-8"), "brewlytics_cafe_comparison.csv", "text/csv")
    st.markdown('<div class="section-title">Affordability vs rating</div>', unsafe_allow_html=True)
    plot = cafe.dropna(subset=["Average_Rating","Listed_Cost_For_Two"])
    if len(plot):
        fig = px.scatter(plot.head(100), x="Listed_Cost_For_Two", y="Average_Rating", size="Review_Records", color="City", hover_name="Name", hover_data=["Review_Records","Cuisine_Tags"], size_max=26, labels={"Listed_Cost_For_Two":"Listed cost for two (₹)","Average_Rating":"Average listed rating"})
        st.plotly_chart(chart_style(fig, 430), use_container_width=True)
    st.caption("This comparison uses listing-level prices and ratings. It does not measure sales, demand or profitability.")

with tabs[3]:
    st.markdown('<div class="section-title">Machine learning lab</div>', unsafe_allow_html=True)
    st.write("A TF-IDF + Logistic Regression classifier predicts rating-derived review classes from text. Since target labels come from star ratings, this is a learning demonstration—not verified sentiment ground truth.")
    training = df[df["Review"].str.len().gt(8) & df["Overall_Rating"].notna()].copy()
    training["Target"] = training["Overall_Rating"].apply(lambda r:"Positive" if r>=4 else ("Negative" if r<=2.5 else "Neutral"))
    if len(training)>=30 and training["Target"].value_counts().min()>=2 and training["Target"].nunique()>=2:
        try:
            model, y_test, y_pred, acc = train_model(tuple(training["Review"].tolist()), tuple(training["Target"].tolist()))
            p,r,fs,_ = precision_recall_fscore_support(y_test,y_pred,average="weighted",zero_division=0)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Model", "TF-IDF + LR")
            m2.metric("Holdout accuracy", f"{acc:.1%}")
            m3.metric("Weighted F1", f"{fs:.3f}")
            m4.metric("Evaluation rows", f"{len(y_test):,}")
            st.caption("Metrics compare predictions with labels derived from ratings on a stratified holdout split. They do not establish human sentiment accuracy or business outcomes.")
            inp = st.text_area("Test a review", placeholder="Example: Friendly staff, good coffee and comfortable seating.", height=100)
            if st.button("Analyse review", type="primary"):
                if inp.strip():
                    pred = model.predict([inp])[0]
                    probs = model.predict_proba([inp])[0]
                    cls = list(model.named_steps["classifier"].classes_)
                    confidence = float(probs[cls.index(pred)])
                    st.success(f"Predicted rating-derived class: **{pred}** · model probability {confidence:.0%}")
                else: st.warning("Enter a review before analysing it.")
            left,right = st.columns(2)
            with left:
                report = classification_report(y_test,y_pred,output_dict=True,zero_division=0)
                st.markdown("**Precision, recall and F1 by class**")
                st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
            with right:
                labels = sorted(pd.Series(y_test).unique())
                cm = confusion_matrix(y_test,y_pred,labels=labels)
                fig = px.imshow(cm,x=labels,y=labels,text_auto=True,aspect="auto",color_continuous_scale=[[0,"#f3f6f0"],[1,"#214e3b"]],labels={"x":"Predicted class","y":"Actual proxy class","color":"Records"})
                st.plotly_chart(chart_style(fig, 360), use_container_width=True)
        except Exception as exc:
            st.warning(f"The model could not be trained on this dataset split: {exc}")
    else:
        st.warning("There are not enough examples in each rating-derived class to train a stratified evaluation. Check the dataset labels.")

with tabs[4]:
    st.markdown('<div class="section-title">Data quality & downloads</div>', unsafe_allow_html=True)
    quality = pd.DataFrame({"Field":df.columns,"Missing values": [int(df[c].isna().sum()) for c in df.columns],"Unique values":[int(df[c].nunique(dropna=True)) for c in df.columns],"Data type":[str(df[c].dtype) for c in df.columns]})
    q1,q2,q3 = st.columns(3)
    q1.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    q2.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    q3.metric("Columns", f"{len(df.columns):,}")
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download filtered reviews", f.to_csv(index=False).encode("utf-8"), "brewlytics_filtered_reviews.csv", "text/csv")
    st.download_button("⬇ Download complete cleaned dataset", df.to_csv(index=False).encode("utf-8"), "brewlytics_cleaned_dataset.csv", "text/csv")
    st.markdown("**Derived fields and methodology**")
    st.markdown("- **Price tier:** groups listed cost for two into affordability bands.\n- **Rating-derived class:** positive ≥4.0, negative ≤2.5, otherwise neutral.\n- **Review length / cuisine count:** basic descriptive features.\n- **Model:** TF-IDF n-grams + Logistic Regression with stratified holdout evaluation.\n- **Business insights:** describe the filtered dataset; they are not audited market-share or financial estimates.")
    st.markdown('<div class="warning-note"><b>Data limitations:</b> no reliable review dates, customer IDs, POS transactions, actual sales, ingredient costs, rent, wages or operating expenses are present. Therefore daily customer counts, daily expenses, recurring expenses, profit, profit percentage and 2022–2026 trends cannot be calculated from this file.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="small">Brewlytics · Portfolio project · Python · Pandas · Plotly · Streamlit · Scikit-learn</p>', unsafe_allow_html=True)
