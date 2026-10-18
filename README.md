# ☕ Brewlytics — Indian Café Review Intelligence

An interactive Streamlit + machine learning portfolio project built from the supplied `reviews.csv` dataset. It explores Indian café ratings, listed price-for-two, cuisine tags and customer review text.

## What it does

- Interactive dashboard filters: city, listed price tier, rating range, rating-derived sentiment, cuisine tags, review length and text search.
- KPIs: review records in the current view, distinct café names, average listed rating and median listed cost for two.
- Visualisations: rating distribution, price tiers, city comparison, cuisine tags and café comparison.
- Café explorer with CSV export.
- Experimental review text classifier using TF-IDF + Logistic Regression.
- Data-quality panel and downloads of filtered/cleaned data.
- Derived fields: `Price_Tier`, `Rating_Band`, `Sentiment_Label`, `Review_Length`, `Cuisine_Count`, `Review_Opportunity_Score`.

## Important data limitations

The supplied dataset has café name, rating, cuisine, listed cost for two, city and review text. It has **no transaction dates, customer IDs, item quantities, actual sales, cost of goods sold, rent, salaries or other operating expenses**.

Therefore this project does **not** calculate actual daily sales, unique customers, real profit, profit margin or 2022–2026 trends. `Rate for two` is a listed price estimate, not revenue. The sentiment labels are weak labels derived from star ratings (positive >= 4.0, negative <= 2.5, otherwise neutral); the classifier is not trained on human-annotated sentiment. The opportunity score is only a transparent exploratory heuristic based on rating and affordability.

To add genuine financial analytics later, obtain dated point-of-sale transactions and expense records from a participating café, or use a public transaction dataset and clearly document its provenance.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open at the local URL printed in the terminal.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository, for example `brewlytics-indian-cafe-intelligence`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `data/` folder.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit installs the packages listed in `requirements.txt`.
6. Add the public app URL to the GitHub repository's About section and README.

## GitHub upload (browser method)

1. Go to https://github.com/new and create a **Public** repository named `brewlytics-indian-cafe-intelligence`.
2. Add a short description: `Indian café review analytics and rating-derived sentiment ML dashboard using Streamlit`.
3. Click **Add file → Upload files**.
4. Upload the project files while preserving this structure:

```text
brewlytics-indian-cafe-intelligence/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── cafe_reviews_cleaned.csv
```

5. Commit the files.
6. Follow the Streamlit deployment steps above to publish a live dashboard.

## Suggested CV entry

**Brewlytics — Indian Café Review Intelligence & Sentiment ML Dashboard**
Built an interactive Streamlit dashboard to explore 775 café review records across Indian cities, compare listed ratings and price bands, and analyse cuisine and review patterns. Developed a TF-IDF + Logistic Regression text classifier using rating-derived weak labels, with holdout evaluation and data-quality reporting. Implemented filters, Plotly visualisations and CSV exports.

Only retain the row count in your CV if it matches the final data you publish.

## Suggested next improvements

- Add a source citation and date of retrieval for the dataset.
- Validate sentiment with a manually labelled review sample.
- Add an actual review date if a reliable source provides one.
- For profit analysis, add real sales and expense tables; do not infer profit from ratings or listed price.
