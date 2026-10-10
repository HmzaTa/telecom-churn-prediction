import joblib
import pytest
from customers import HIGH_RISK, LOW_RISK
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client(tmp_path, monkeypatch, model_bundle):
    model_path = tmp_path / "churn_model.pkl"
    joblib.dump(model_bundle, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(app) as test_client:  # the `with` block runs the startup (model loading)
        yield test_client


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pkl"))
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=HIGH_RISK)

    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"churn_probability", "prediction", "threshold"}
    assert 0 <= body["churn_probability"] <= 1
    assert body["prediction"] in ("High Risk", "Low Risk")
    assert body["threshold"] == 0.5


def test_high_risk_profile_scores_higher_than_low_risk(client):
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]

    assert high > low


def test_total_charges_is_optional(client):
    payload = {k: v for k, v in HIGH_RISK.items() if k != "TotalCharges"}

    with_default = client.post("/predict", json=payload)
    explicit = client.post("/predict", json=HIGH_RISK)  # 95.5 x 2 = 191.0

    assert with_default.status_code == 200
    assert with_default.json() == explicit.json()


def test_threshold_query_parameter(client):
    always_high = client.post("/predict?threshold=0", json=LOW_RISK).json()
    always_low = client.post("/predict?threshold=1", json=HIGH_RISK).json()

    assert always_high["prediction"] == "High Risk"
    assert always_low["prediction"] == "Low Risk"


@pytest.mark.parametrize("change", [
    {"Contract": "Weekly"},        # category the model has never seen
    {"tenure": -1},                # negative tenure
    {"MonthlyCharges": -5.0},      # negative charge
    {"SeniorCitizen": 2},          # not 0 or 1
    {"unexpected_field": "x"},     # typo or extra field
])
def test_invalid_input_is_rejected(client, change):
    response = client.post("/predict", json={**HIGH_RISK, **change})

    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    payload = {k: v for k, v in HIGH_RISK.items() if k != "Contract"}

    assert client.post("/predict", json=payload).status_code == 422


def test_threshold_out_of_range_is_rejected(client):
    assert client.post("/predict?threshold=1.5", json=HIGH_RISK).status_code == 422


def test_api_reports_missing_model(client_without_model):
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json=HIGH_RISK).status_code == 503
