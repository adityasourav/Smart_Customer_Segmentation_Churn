import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


def train_segmentation_model():

    df = pd.read_csv(
        "data/Customer-Churn.csv"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    X = df[FEATURES].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(
        X_scaled
    )

    joblib.dump(
        model,
        "notebook/models/segmentation_model.pkl"
    )

    joblib.dump(
        scaler,
        "notebook/models/segmentation_scaler.pkl"
    )

    return df, model, scaler


if __name__ == "__main__":

    df, model, scaler = train_segmentation_model()

    print("Segmentation model trained successfully.")
    print()

    print("Cluster Distribution:")
    print(df["Cluster"].value_counts().sort_index())

    print()

    print("Cluster Profiles:")

    print(
        df.groupby("Cluster")[FEATURES].mean().round(2)
    )