import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page configuration
st.set_page_config(page_title="K-Means Clustering Framework with Segmentation", layout="wide")

# Sidebar - Logo and Developers
st.sidebar.image(
    "https://www.asean-competition.org/file/post_image/LCyh3I_post_MyCC.jpg",
    use_container_width=True
)
st.sidebar.header("Developers:")
for dev in ["Ku Muhammad Naim Ku Khalif"]:
    st.sidebar.write(f"- {dev}")

# Main Title
st.title("K-Means Clustering Framework with Segmentation")

# Sidebar - Upload data and select features
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    features = st.sidebar.multiselect("Select feature columns", df.columns)
    if features:
        X = df[features]

        # Elbow method parameters
        st.sidebar.subheader("Elbow Method Parameters")
        max_k = st.sidebar.slider("Max number of clusters to try for Elbow Method", 2, 15, 10)

        # Compute WCSS for range of k values
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Plot Elbow Method curve
        st.subheader("Elbow Method to help find optimal k")
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(range(1, max_k + 1), wcss, marker='o')
        ax_elbow.set_xlabel('Number of clusters k')
        ax_elbow.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        ax_elbow.set_title('Elbow Method For Optimal k')
        st.pyplot(fig_elbow)

        # Choose number of clusters from slider (default 3)
        n_clusters = st.sidebar.slider("Number of Clusters (k) for KMeans", 2, max_k, 3)

        # Model training
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)

        df_clusters = X.copy()
        df_clusters['Cluster'] = clusters

        # Performance metric
        sil_score = silhouette_score(X, clusters)
        st.subheader("Clustering Performance")
        st.write(f"Silhouette Score: {sil_score:.3f}")

        # Cluster centers
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
        st.subheader("Cluster Centers")
        st.write(centers)

        # Plot clusters (first two features)
        if len(features) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=df_clusters,
                x=features[0], y=features[1],
                hue='Cluster', palette='tab10', ax=ax
            )
            ax.set_title(f'Clusters by {features[0]} and {features[1]}')
            st.pyplot(fig)

        # Predict cluster for new input
        st.subheader("Assign New Data to Cluster")
        input_vals = []
        for feat in features:
            val = st.number_input(f"Input value for {feat}", value=float(df[feat].mean()))
            input_vals.append(val)
        if st.button("Assign Cluster"):
            input_array = np.array(input_vals).reshape(1, -1)
            pred_cluster = kmeans.predict(input_array)[0]
            st.success(f"Assigned to Cluster: {pred_cluster}")

    else:
        st.info("Please select at least one feature column.")
else:
    st.info("Upload a CSV file to start.")
