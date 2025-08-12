import os
import json
import glob
import argparse
import pandas as pd
import joblib

from src.feature_engineering import AdvancedFeatureEngineering


def _find_latest_model_info(models_dir: str = "models/automl") -> str:
    """Find the latest model_info_*.json by timestamp in filename."""
    pattern = os.path.join(models_dir, "model_info_*.json")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No model_info_*.json found under {models_dir}")
    # Candidates already sorted lexicographically which works with YYYYMMDD_HHMMSS
    return candidates[-1]


def load_artifacts(models_dir: str = "models/automl") -> dict:
    """Load latest model, scaler and features artifact paths from model_info JSON."""
    info_path = _find_latest_model_info(models_dir)
    with open(info_path, "r") as f:
        info = json.load(f)

    model_path = info.get("model_path")
    scaler_path = info.get("scaler_path")
    features_path = info.get("features_path")

    if not (model_path and scaler_path and features_path):
        raise ValueError("Model info JSON missing required paths (model_path/scaler_path/features_path)")

    # Make absolute if needed
    model_path = os.path.join(models_dir, os.path.basename(model_path)) if not os.path.isabs(model_path) else model_path
    scaler_path = os.path.join(models_dir, os.path.basename(scaler_path)) if not os.path.isabs(scaler_path) else scaler_path
    features_path = os.path.join(models_dir, os.path.basename(features_path)) if not os.path.isabs(features_path) else features_path

    return {
        "info_path": info_path,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "features_path": features_path,
        "meta": info,
    }


def predict_all(
    data_path: str = "data/birlesik_risk_verisi.csv",
    output_path: str = "reports/predictions_all.csv",
    models_dir: str = "models/automl",
) -> pd.DataFrame:
    """Run batch predictions for all rows in data_path using latest saved AutoML artifacts."""

    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    raw_df = pd.read_csv(data_path)

    # Load artifacts
    artifacts = load_artifacts(models_dir)
    model = joblib.load(artifacts["model_path"])
    scaler = joblib.load(artifacts["scaler_path"])
    feature_names = joblib.load(artifacts["features_path"])  # List[str]

    # Feature engineering (must match training)
    fe = AdvancedFeatureEngineering()
    df = fe.create_advanced_features(raw_df)

    # Align features
    missing_features = [c for c in feature_names if c not in df.columns]
    if missing_features:
        # Create missing columns with zeros to keep schema
        for c in missing_features:
            df[c] = 0.0

    X = df[feature_names].fillna(0)
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)
    preds = pd.Series(preds).clip(0, 100)

    # Build output
    out = pd.DataFrame({
        "ProjectId": raw_df.get("ProjectId", pd.Series(range(len(df)))).values,
        "PredictedRiskScore": preds.values,
    })

    # Risk category bins (ensure 0 included)
    from src.risk_calculator import get_risk_category
    out["RiskCategory"] = get_risk_category(out["PredictedRiskScore"])

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    return out


def main():
    parser = argparse.ArgumentParser(description="Batch risk prediction for all customers")
    parser.add_argument("--data", dest="data_path", default="data/birlesik_risk_verisi.csv")
    parser.add_argument("--out", dest="output_path", default="reports/predictions_all.csv")
    parser.add_argument("--models", dest="models_dir", default="models/automl")
    args = parser.parse_args()

    df_out = predict_all(args.data_path, args.output_path, args.models_dir)
    # Print brief summary
    print(f"✅ Saved predictions: {args.output_path}  (rows={len(df_out)})")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


