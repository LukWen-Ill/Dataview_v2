from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from src.models import RANDOM_STATE



PRICE_FEATURES = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

def prepare_pricing_data(df):
    """Gör en pricing på vilket pris dem får beroende på vilka tjänster kunden har"""
    internet = df[df["InternetService"] != "No"]
    X = internet[PRICE_FEATURES]
    y = internet["MonthlyCharges"]
    return X, y

def train_pricing_model(X, y, random_state=RANDOM_STATE):
    """Tränar modellen för att kunna prediktera vad dem kommer få för pris"""
    X_trainVal, X_test, y_trainVal, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("regressor", LinearRegression())
    ])
    pipeline.fit(X_trainVal, y_trainVal)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return pipeline, r2

def price_list(pipeline) -> pd.DataFrame:
    """Vad varje tjänst kostar per månad, enligt modellens koefficienter"""
    names = pipeline.named_steps["onehot"].get_feature_names_out()
    coefs = pipeline.named_steps["regressor"].coef_
    return pd.DataFrame({"tjanst": names, "USD_per_monad": coefs}).sort_values("USD_per_monad", ascending=False)