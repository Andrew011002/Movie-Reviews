import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import process_functions as pf

path = pf.path

df = pd.read_csv(f"{path}/data.csv", names=["Reviews", "Sentiment"])
df = df.drop(0, axis=0)

size = 50000

transformer = TFIDF()

print(f"Loading in the Data...")

data = pf.clean(df["Reviews"].values[:size])

transformer.fit(data)

features = np.array(transformer.transform(data).toarray())
labels = np.array([1 if l == "positive" else 0 for l in df["Sentiment"].values[:size]])

print(f"Data Loaded...")

