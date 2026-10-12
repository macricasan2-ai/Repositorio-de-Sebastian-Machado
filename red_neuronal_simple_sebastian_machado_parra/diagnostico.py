import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("heart_disease.csv")
y = df["Heart Disease Status"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(df.drop(columns="Heart Disease Status"), dummy_na=True, dtype=int)
X = X.fillna(X.median())

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modelos = [
    ("Regresión logística", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
]
for nombre, modelo in modelos:
    modelo.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1])
    print(f"{nombre}: AUC = {auc:.3f}")