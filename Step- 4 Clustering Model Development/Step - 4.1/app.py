
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Buyer Intelligence")
st.caption(
    "Machine Learning-Based Buyer Segmentation "
    "and Investment Profiling"
)

# --------------------------------------------------
# 2. FILE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_csv(filename):
    return pd.read_csv(BASE_DIR / filename)

@st.cache_resource
def load_model(filename):
    return joblib.load(BASE_DIR / filename)

@st.cache_data
def load_config(filename):
    with open(BASE_DIR / filename, "r") as file:
        return json.load(file)

# --------------------------------------------------
# 3. LOAD PROJECT DATA
# --------------------------------------------------

try:
    clients = load_csv("clients_cleaned.csv")
    properties = load_csv("properties_cleaned.csv")
    assignments = load_csv("weighted_kmeans_k2_assignments.csv")
    profiles = load_csv("final_buyer_cluster_profiles.csv")

    model = load_model(
        "weighted_kmeans_buyer_segmentation_model.joblib"
    )
    preprocessor = load_model(
        "buyer_segmentation_preprocessor.joblib"
    )
    config = load_config("weighted_clustering_config.json")

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()

# --------------------------------------------------
# 4. MERGE CUSTOMER ASSIGNMENTS
# --------------------------------------------------

data = clients.merge(
    assignments[["client_id", "cluster"]],
    on="client_id",
    how="left",
    validate="one_to_one"
)

data["cluster"] = data["cluster"].astype("Int64")

# Calculate age using the same reference date as training
dob = pd.to_datetime(data["date_of_birth"], errors="coerce")

data["age"] = (
    2025
    - dob.dt.year
    - (
        (dob.dt.month > 1)
        | (
            (dob.dt.month == 1)
            & (dob.dt.day > 1)
        )
    ).astype("Int64")
)

# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Overview",
        "Buyer Analytics",
        "Customer Explorer",
        "Model Information"
    ]
)

# --------------------------------------------------
# 6. OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.subheader("Market Intelligence Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{len(data):,}")
    col2.metric(
        "Cluster 0 Customers",
        f"{(data['cluster'] == 0).sum():,}"
    )
    col3.metric(
        "Cluster 1 Customers",
        f"{(data['cluster'] == 1).sum():,}"
    )
    col4.metric("Total Properties", f"{len(properties):,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        cluster_counts = (
            data["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            cluster_counts,
            names="Cluster",
            values="Customers",
            title="Buyer Cluster Distribution",
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "acquisition_purpose" in data.columns:
            purpose_counts = (
                data["acquisition_purpose"]
                .value_counts()
                .rename_axis("Purpose")
                .reset_index(name="Customers")
            )

            fig = px.bar(
                purpose_counts,
                x="Purpose",
                y="Customers",
                title="Customer Acquisition Purpose",
                text="Customers"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Buyer Cluster Profiles")
    st.dataframe(profiles, use_container_width=True)

# --------------------------------------------------
# 7. BUYER ANALYTICS
# --------------------------------------------------

elif page == "Buyer Analytics":

    st.subheader("Buyer Profile Analysis")

    selected_cluster = st.selectbox(
        "Select cluster",
        sorted(data["cluster"].dropna().unique().tolist())
    )

    subset = data[data["cluster"] == selected_cluster]

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers", len(subset))
    col2.metric(
        "Average Age",
        f"{subset['age'].mean():.1f} years"
    )
    col3.metric(
        "Average Satisfaction",
        f"{subset['satisfaction_score'].mean():.2f}"
    )

    if "acquisition_purpose" in subset.columns:
        fig = px.histogram(
            subset,
            x="acquisition_purpose",
            title="Acquisition Purpose",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

    if "loan_applied" in subset.columns:
        loan_data = (
            subset["loan_applied"]
            .value_counts()
            .rename_axis("Loan Applied")
            .reset_index(name="Customers")
        )

        fig = px.bar(
            loan_data,
            x="Loan Applied",
            y="Customers",
            title="Loan Application Distribution",
            text="Customers"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Cluster numbers are unsupervised group labels, "
        "not investment-quality scores."
    )

# --------------------------------------------------
# 8. CUSTOMER EXPLORER
# --------------------------------------------------

elif page == "Customer Explorer":

    st.subheader("Explore Segmented Customers")

    cluster_filter = st.selectbox(
        "Filter by cluster",
        ["All"] + sorted(
            data["cluster"].dropna().unique().tolist()
        )
    )

    filtered = data.copy()

    if cluster_filter != "All":
        filtered = filtered[
            filtered["cluster"] == cluster_filter
        ]

    search_id = st.text_input("Search customer ID")

    if search_id:
        filtered = filtered[
            filtered["client_id"]
            .astype(str)
            .str.contains(search_id, case=False, na=False)
        ]

    st.write(f"Matching customers: {len(filtered):,}")

    display_columns = [
        column for column in [
            "client_id",
            "cluster",
            "age",
            "client_type",
            "gender",
            "country",
            "region",
            "acquisition_purpose",
            "loan_applied",
            "satisfaction_score"
        ]
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Customer Results",
        data=csv_data,
        file_name="buyer_segmentation_results.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 9. MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.subheader("Model Information")

    st.write("Algorithm: Weighted K-Means Clustering")
    st.write("Number of clusters:", config["n_clusters"])
    st.write("Random state:", config["random_state"])
    st.write("Expected model features:", len(config["features"]))

    st.write("Excluded features:", config["excluded_features"])

    st.info(
        "The model groups customers by feature similarity. "
        "Cluster assignments do not establish investment returns, "
        "creditworthiness, or future property performance."
    )

    st.subheader("Saved Cluster Profiles")
    st.dataframe(profiles, use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Buyer Segmentation Project")
