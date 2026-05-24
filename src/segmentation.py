from sklearn.cluster import KMeans

def create_clusters(df):

    model = KMeans(
        n_clusters=4,
        random_state=42
    )

    df["Cluster"] = model.fit_predict(
        df[
            ["tenure",
             "MonthlyCharges",
             "TotalCharges"]
        ]
    )

    return df