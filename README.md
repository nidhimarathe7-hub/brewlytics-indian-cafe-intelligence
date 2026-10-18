# ☕ Brewlytics — Indian Café Review Intelligence

A polished interactive Streamlit dashboard exploring café review records, ratings, cuisine tags and listed cost-for-two values. Built as a Business Analytics / ML portfolio project.

## Features
- Five dashboard sections: executive overview, customer voice, café explorer, ML lab, and data quality/export.
- Combined sidebar filters for city, listed price tier, rating range, rating-derived class, cuisine, listed price range, minimum review length, and free-text search.
- KPIs, market comparisons, price-tier mix, rating distribution, cuisine tags, customer review explorer and café rankings.
- TF-IDF + Logistic Regression text classifier, holdout accuracy, weighted F1, class-level precision/recall, confusion matrix and interactive prediction input.
- Data-quality checks and downloadable filtered review/café comparison CSVs.
- Robust dataset loading supports either `data/cafe_reviews_cleaned.csv` or `cafe_reviews_cleaned.csv` at repository root.

## Important limitations
The dataset contains café listings, ratings, cuisine tags, listed cost for two and review text. It does **not** contain reliable transaction dates, customer IDs, actual sales, product quantities, ingredient costs, rent, wages or other expenses. Consequently, this project does not calculate real daily customers, sales, daily/recurring expenses, profit, profit margin or 2022–2026 trends. Do not infer those fields from listed cost for two.

The model target is derived from star ratings (positive ≥4.0; negative ≤2.5; otherwise neutral), not independently human-annotated sentiment. Evaluation metrics describe agreement with these proxy labels, not true emotion detection. City and café record counts represent dataset coverage, not market share.

## Run locally
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Commit `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and the dataset.
2. Keep the CSV either at `data/cafe_reviews_cleaned.csv` or at the repository root named `cafe_reviews_cleaned.csv`.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Create an app, select this repository and the `main` branch, and set the main file path to `app.py`.
5. Deploy. After a code commit, Streamlit should redeploy automatically.

## Suggested CV description
**Brewlytics — Indian Café Review Intelligence & ML Dashboard:** Built an interactive Streamlit application to explore café review records across Indian cities, compare ratings and listed price bands, analyse review patterns, and rank cafés under dynamic filters. Implemented a TF-IDF + Logistic Regression classifier with holdout evaluation, confusion matrix, data-quality checks and CSV exports. Documented the limits of rating-derived labels and non-transactional data.
