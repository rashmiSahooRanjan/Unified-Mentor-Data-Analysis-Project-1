# Real Estate Buyer Segmentation & Investment Profiling

Developed a Machine Learning–based Real Estate Buyer Segmentation and Investment Profiling dashboard to analyze customer behavior and identify distinct buyer segments in the real estate market. The project uses a weighted K-Means clustering approach to group buyers based on available demographic, geographic, acquisition-purpose, and loan-related features. An interactive Streamlit dashboard presents cluster distributions, buyer behavior patterns, geographic comparisons, descriptive statistics, and customer-level insights. It also supports filtered customer exploration and CSV export to facilitate data-driven market analysis. The clustering model achieved a silhouette score of 0.3012, providing an initial measure of cluster separation. The project demonstrates how machine learning and interactive data visualization can support real estate market intelligence while acknowledging the limitations of the available data.

## Key Project Highlights

- **Machine Learning**: Applied weighted K-Means clustering with K=2 to segment real estate buyers.
- **Buyer Segmentation**: Analyzed 2,000 buyers across two clusters.
- **Behavioral Analysis**: Examined acquisition purpose, loan application patterns, and referral channels.
- **Interactive Dashboard**: Developed a Streamlit dashboard with dynamic charts, filters, and segment-level insights.
- **Geographic Analysis*: Compared buyer segments across available countries and regions.
- **Customer Explorer**: Enabled filtered customer record viewing and CSV export.
- **Model Evaluation**: Assessed clustering using Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Index.
- **Data-Driven Insights**: Generated descriptive buyer profiles to support real estate market analysis.
- **Responsible ML**: Distinguished descriptive customer segments from unverified assumptions about wealth or investment profitability.

## Dashboard

- Buyer Segmentation Overview: cluster counts and shares.
- Investor Behavior Dashboard: acquisition purpose, loan patterns, and linked property summaries.
- Geographic Buyer Analysis: region-by-cluster charts; maps only when coordinates are supplied.
- Segment Insights Panel: descriptive statistics by cluster.
- Customer Explorer: filtered records and CSV export.
- Methodology & Validation: pipeline and limitations.

## Files needed

Place `clients_cleaned.csv` beside `app.py` or in `data/`. Optional files: `properties_cleaned.csv` and `weighted_kmeans_k2_assignments.csv` (columns `client_id`, `cluster`). Original customer CSVs are not bundled because they were not available in the file-generation workspace. Upload them through the app sidebar or add them to your repository if permitted.

Client columns: `client_id`, `client_type`, `gender`, `country`, `region`, `date_of_birth` or `age`, `acquisition_purpose`, `loan_applied`, `referral_channel`; optional `satisfaction_score`.

Property columns may include `listing_id`, `transaction_date`, `unit_category`, `unit_number`, `floor_area_sqft`, `sale_price`, `listing_status`, `client_ref`. Buyer-property links use `properties.client_ref == clients.client_id`; verify identifier formats.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
## 👨‍💻 Authors

| Name                          | Role      |
| ----------------------------- | --------- |
| **Rashmi Ranjan Sahoo** | Developer |


## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, `README.md` and allowed data files to GitHub.
2. Sign in at https://share.streamlit.io/ with GitHub.
3. Select the repository, branch and `app.py`, then deploy.
4. If data are not committed, upload the client CSV in the app sidebar.
5. Verify counts and charts against the notebook.

## Recorded notebook results

Existing weighted K-Means run used K=2 and 79 features after excluding `satisfaction_score`. Cluster 0: 969 buyers (48.45%); cluster 1: 1,031 buyers (51.55%). Silhouette: 0.3012; Davies–Bouldin: 1.2913; Calinski–Harabasz: 1168.6032. Mean ages: 38.80 and 68.48 years, respectively. These are internal clustering results and descriptive profiles, not ground-truth accuracy or evidence of investment profitability.

## Responsible interpretation

The source schema has no income, net worth, luxury status or investment-return fields. Do not label clusters as high-net-worth/luxury investors without additional evidence. Region names alone do not establish coordinates. Do not publish real customer data to a public repository unless authorized.
