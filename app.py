from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Real Estate Buyer Intelligence", page_icon="🏙️", layout="wide")
ROOT = Path(__file__).parent
st.title("🏙️ Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Unified Mentor / Parcl Co. Limited | Exploratory buyer intelligence")

def read_csv_candidates(names):
    for name in names:
        for p in [ROOT/name, ROOT/"data"/name]:
            if p.exists():
                return pd.read_csv(p), str(p)
    return None, None

def clean_clients(df):
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip().replace({"":pd.NA, "nan":pd.NA, "None":pd.NA})
    if "date_of_birth" in df:
        dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
        df["age"] = ((pd.Timestamp("2025-01-01") - dob).dt.days / 365.2425).round()
        df.loc[(df.age < 18) | (df.age > 110), "age"] = np.nan
    elif "age" not in df:
        df["age"] = np.nan
    if "loan_applied" in df:
        s = df["loan_applied"].astype("string").str.lower().str.strip()
        df["loan_applied"] = s.map({"yes":"Yes","y":"Yes","true":"Yes","1":"Yes","no":"No","n":"No","false":"No","0":"No"}).fillna(df["loan_applied"])
    return df.drop_duplicates("client_id")

def make_clusters(df, k):
    cats = [c for c in ["client_type","gender","country","region","acquisition_purpose","loan_applied","referral_channel"] if c in df]
    nums = [c for c in ["age"] if c in df]
    if not cats and not nums:
        raise ValueError("No usable buyer features were found.")
    transformers = []
    if cats:
        transformers.append(("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]), cats))
    if nums:
        transformers.append(("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), nums))
    prep = ColumnTransformer(transformers, remainder="drop")
    X = prep.fit_transform(df)
    names = prep.get_feature_names_out()
    # Scale categorical feature blocks to reduce high-cardinality domination.
    for col in cats:
        idx = [i for i,n in enumerate(names) if n.startswith(f"cat__{col}_")]
        if idx:
            X[:,idx] /= np.sqrt(len(idx))
    model = KMeans(n_clusters=min(k, max(2,len(df)-1)), random_state=42, n_init=20)
    labels = model.fit_predict(X)
    metrics = {}
    if len(np.unique(labels)) > 1 and len(df) > len(np.unique(labels)):
        sample_idx = np.arange(len(df)) if len(df) <= 5000 else np.random.RandomState(42).choice(len(df),5000,replace=False)
        metrics = {"Silhouette score":round(float(silhouette_score(X[sample_idx], labels[sample_idx])),4),
                   "Davies-Bouldin index":round(float(davies_bouldin_score(X,labels)),4)}
    return labels.astype(str), metrics

with st.sidebar:
    st.header("Upload data")
    up_clients = st.file_uploader("clients_cleaned.csv (required)", type="csv")
    up_properties = st.file_uploader("properties_cleaned.csv (optional)", type="csv")
    k = st.slider("Clusters if calculating in app", 2, 6, 2)
    st.caption("Filters apply to all dashboard pages.")

clients, client_source = read_csv_candidates(["clients_cleaned.csv","clients.csv"])
properties, property_source = read_csv_candidates(["properties_cleaned.csv","properties.csv"])
assignments, assignment_source = read_csv_candidates(["weighted_kmeans_k2_assignments.csv"])
if up_clients: clients, client_source = pd.read_csv(up_clients), "uploaded client CSV"
if up_properties: properties, property_source = pd.read_csv(up_properties), "uploaded property CSV"
if clients is None:
    st.info("Upload `clients_cleaned.csv` in the sidebar or place it beside app.py. The original customer data is not included in this ZIP.")
    st.markdown("Required columns: `client_id`, `client_type`, `gender`, `country`, `region`, `date_of_birth` (or `age`), `acquisition_purpose`, `loan_applied`, `referral_channel`. `satisfaction_score` is optional.")
    st.stop()
if "client_id" not in clients:
    st.error("Client data must include client_id.")
    st.stop()
clients = clean_clients(clients)
clients["client_id"] = clients["client_id"].astype(str)
if assignments is not None and {"client_id","cluster"}.issubset(assignments.columns):
    assignments["client_id"] = assignments["client_id"].astype(str)
    data = clients.merge(assignments[["client_id","cluster"]], on="client_id", how="left")
else:
    data = clients.copy()
    data["cluster"] = np.nan
if data["cluster"].isna().any():
    try:
        labels, metrics = make_clusters(clients, k)
        data["cluster"] = labels
        model_status = f"Calculated K-Means (K={k})"
    except Exception as e:
        st.error(f"Could not calculate clusters: {e}")
        st.stop()
else:
    data["cluster"] = data["cluster"].astype(str)
    metrics = {}
    model_status = f"Loaded saved assignments: {assignment_source}"

with st.sidebar:
    filtered = data.copy()
    for col in ["country","region","acquisition_purpose","client_type"]:
        if col in filtered:
            opts = sorted(filtered[col].dropna().astype(str).unique())
            choice = st.multiselect(col.replace("_"," ").title(), opts)
            if choice: filtered = filtered[filtered[col].astype(str).isin(choice)]
    page = st.radio("Dashboard", ["Buyer Segmentation Overview","Investor Behavior Dashboard","Geographic Buyer Analysis","Segment Insights Panel","Customer Explorer","Methodology & Validation"])

m1,m2,m3,m4 = st.columns(4)
m1.metric("Filtered buyers", f"{len(filtered):,}")
m2.metric("Clusters", str(filtered.cluster.nunique()))
m3.metric("Countries", str(filtered.country.nunique()) if "country" in filtered else "—")
m4.metric("Regions", str(filtered.region.nunique()) if "region" in filtered else "—")
st.caption(f"Source: {client_source} | {model_status}")

if page == "Buyer Segmentation Overview":
    st.subheader("Buyer Segmentation Overview")
    counts = filtered.cluster.astype(str).value_counts().sort_index().rename_axis("Cluster").reset_index(name="Buyers")
    if len(counts):
        counts["Share (%)"] = counts.Buyers/counts.Buyers.sum()*100
        a,b = st.columns(2)
        a.plotly_chart(px.bar(counts,x="Cluster",y="Buyers",text="Buyers",title="Buyer count by cluster"),use_container_width=True)
        b.plotly_chart(px.pie(counts,names="Cluster",values="Buyers",hole=.45,title="Cluster share"),use_container_width=True)
        st.dataframe(counts.round(2),use_container_width=True)
        st.download_button("Download filtered buyers",filtered.to_csv(index=False).encode(),"filtered_buyers.csv","text/csv")

elif page == "Investor Behavior Dashboard":
    st.subheader("Investor Behavior Dashboard")
    for col, title in [("acquisition_purpose","Acquisition purpose by cluster"),("loan_applied","Loan application patterns by cluster")]:
        if col in filtered:
            tmp=filtered.groupby(["cluster",col],dropna=False).size().reset_index(name="Buyers")
            tmp["Share within cluster (%)"]=100*tmp.Buyers/tmp.groupby("cluster").Buyers.transform("sum")
            st.plotly_chart(px.bar(tmp,x="cluster",y="Share within cluster (%)",color=col,barmode="stack",title=title),use_container_width=True)
            st.dataframe(tmp,use_container_width=True)
    if properties is not None:
        prop=properties.copy()
        if {"client_ref","listing_status"}.issubset(prop.columns):
            prop["client_ref"]=prop.client_ref.astype(str)
            link=data[["client_id","cluster"]].copy()
            prop=prop.merge(link,left_on="client_ref",right_on="client_id",how="inner")
            prop=prop[prop.client_id.isin(filtered.client_id)]
            if "sale_price" in prop:
                prop["sale_price"]=pd.to_numeric(prop.sale_price,errors="coerce")
                st.dataframe(prop.groupby("cluster").sale_price.agg(["count","mean","median","sum"]).round(2),use_container_width=True)
                st.plotly_chart(px.box(prop,x="cluster",y="sale_price",color="cluster",title="Recorded sale price by linked buyer cluster"),use_container_width=True)
            st.plotly_chart(px.histogram(prop,x="cluster",color="listing_status",barmode="group",title="Listing status by linked buyer cluster"),use_container_width=True)
        else:
            st.info("Property file needs client_ref and listing_status. Property records are analyzed only when buyer links are available.")
    else:
        st.info("Upload properties_cleaned.csv to enable property-level analysis.")

elif page == "Geographic Buyer Analysis":
    st.subheader("Geographic Buyer Analysis")
    if "region" in filtered:
        region=filtered.groupby(["region","cluster"],dropna=False).size().reset_index(name="Buyers")
        top=filtered.region.value_counts().head(25).index
        region=region[region.region.isin(top)]
        st.plotly_chart(px.bar(region,x="region",y="Buyers",color="cluster",barmode="stack",title="Segments in the 25 most represented regions"),use_container_width=True)
        st.plotly_chart(px.density_heatmap(region,x="region",y="cluster",z="Buyers",histfunc="sum",title="Region-by-cluster concentration"),use_container_width=True)
        st.dataframe(region.sort_values("Buyers",ascending=False),use_container_width=True)
        if {"latitude","longitude"}.issubset(filtered.columns):
            geo=filtered.dropna(subset=["latitude","longitude"])
            if len(geo): st.plotly_chart(px.scatter_mapbox(geo,lat="latitude",lon="longitude",color="cluster",hover_name="region",mapbox_style="open-street-map",zoom=1),use_container_width=True)
        else:
            st.caption("The source schema provides region labels, not coordinates. A true map is shown only if latitude and longitude are supplied; this dashboard does not invent coordinates.")
    else: st.warning("No region column was found.")

elif page == "Segment Insights Panel":
    st.subheader("Segment Insights Panel")
    rows=[]
    for cl,g in filtered.groupby("cluster",dropna=False):
        row={"Cluster":str(cl),"Buyers":len(g),"Share (%)":100*len(g)/max(1,len(filtered))}
        for col in ["age","satisfaction_score"]:
            if col in g:
                row[f"Mean {col}"]=pd.to_numeric(g[col],errors="coerce").mean()
                row[f"Median {col}"]=pd.to_numeric(g[col],errors="coerce").median()
        for col in ["acquisition_purpose","client_type","loan_applied","gender","country","region","referral_channel"]:
            if col in g and len(g[col].dropna()):
                vc=g[col].astype("string").fillna("Missing").value_counts()
                row[f"Top {col}"]=f"{vc.index[0]} ({vc.iloc[0]/len(g)*100:.1f}%)"
        rows.append(row)
    summary=pd.DataFrame(rows)
    st.dataframe(summary.round(2),use_container_width=True)
    if len(summary): st.download_button("Download segment summary",summary.to_csv(index=False).encode(),"segment_insights.csv","text/csv")

elif page == "Customer Explorer":
    st.subheader("Customer Explorer")
    cols=[c for c in ["client_id","cluster","client_type","gender","age","country","region","acquisition_purpose","loan_applied","referral_channel","satisfaction_score"] if c in filtered]
    st.dataframe(filtered[cols],use_container_width=True,height=480)
    st.download_button("Export filtered customers",filtered.to_csv(index=False).encode(),"customer_explorer.csv","text/csv")

else:
    st.subheader("Methodology & Validation")
    st.markdown("""
    1. Trim categorical text and deduplicate by client_id.
    2. Derive age using 1 January 2025 as a fixed reference date.
    3. Impute missing values, one-hot encode categorical variables and standardize age.
    4. Scale each categorical feature block to reduce high-cardinality dominance.
    5. Apply K-Means with fixed random state; use saved assignments when supplied.

    **Interpretation limits:** clustering is unsupervised; there are no ground-truth segment labels. Silhouette is not accuracy. The dataset does not contain income, net worth or property-return outcomes. Region names do not provide map coordinates. Cluster IDs are arbitrary.
    """)
    if metrics: st.json(metrics)
    st.dataframe(pd.DataFrame({"Check":["Rows","Unique client IDs","Missing IDs","Missing region values"],"Result":[len(clients),clients.client_id.nunique(),int(clients.client_id.isna().sum()),int(clients.region.isna().sum()) if "region" in clients else "N/A"]}),use_container_width=True)

with st.expander("About"):
    st.write("Exploratory decision-support prototype. It is not property valuation, investment advice, or a verified prediction of buyer intent.")
