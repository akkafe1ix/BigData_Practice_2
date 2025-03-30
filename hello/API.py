#python -m uvicorn API:app --reload --host 127.0.0.1 --port 8001

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Custom KNN ---
class CustomKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

# --- Глобальные переменные ---
df = None
models = {}
X_train = X_val = y_train = y_val = None
X_scaled = y_all = None
selected_features = ["gravity", "ph"]

@app.on_event("startup")
def train_models():
    global df, X_train, X_val, y_train, y_val, X_scaled, y_all, models

    df = pd.read_csv("data/train.csv")
    X = df[selected_features].values
    y = df["target"].values
    y_all = y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "CustomKNN": CustomKNN(k=5),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(),
        "SVM": SVC(probability=True)
    }

    for model in models.values():
        model.fit(X_train, y_train)

@app.get("/data_info")
def get_data_info():
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "head": df.head(10).to_dict(orient="records")
    }

@app.get("/eda")
def get_eda():
    numerical = df.select_dtypes(include=["int64", "float64"])
    stats = numerical.agg(["min", "median", "mean", "max"]).T
    stats["25%"] = numerical.apply(lambda x: np.percentile(x, 25))
    stats["75%"] = numerical.apply(lambda x: np.percentile(x, 75))
    stats = stats.round(3).reset_index().rename(columns={"index": "feature"})

    cat_features = df.select_dtypes(include=["object"]).columns.tolist()
    cat_data = {}
    for col in cat_features:
        top_val = df[col].mode()[0] if not df[col].mode().empty else None
        freq = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        cat_data[col] = {"mode": top_val, "frequency": freq}

    return {
        "numerical": stats.to_dict(orient="records"),
        "categorical": cat_data
    }

@app.get("/metrics")
def get_metrics():
    result = []
    for name, model in models.items():
        y_pred = model.predict(X_val)
        try:
            roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]) if hasattr(model, "predict_proba") else None
        except:
            roc_auc = None
        result.append({
            "name": name,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc
        })
    return result

@app.get("/roc")
def get_roc():
    roc_data = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        else:
            probs = model.predict(X_val)
        fpr, tpr, _ = roc_curve(y_val, probs)
        roc_data.append({
            "name": name,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        })
    return roc_data

@app.get("/boundaries")
def get_boundaries():
    from matplotlib.colors import ListedColormap
    boundaries = {}
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    for name, model in models.items():
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)
        boundaries[name] = {
            "xx": xx.tolist(),
            "yy": yy.tolist(),
            "Z": Z.tolist()
        }
    return {
        "boundaries": boundaries,
        "X": X_scaled.tolist(),
        "y": y_all.tolist()
    }
