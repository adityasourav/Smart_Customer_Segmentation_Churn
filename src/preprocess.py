import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(path):

    df=pd.read_csv(path)

    df.drop_duplicates(inplace=True)

    encoder=LabelEncoder()

    for col in df.columns:

        if df[col].dtype=="object":

            df[col]=encoder.fit_transform(df[col])

    return df