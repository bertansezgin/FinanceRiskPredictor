import pandas as pd
import joblib
from src.preprocessing import clean_data, generate_features

def predict_risk(csv_path="data/yeni_musteri.csv", model_path="models/linear_model.pkl"):
    # Yeni veriyi oku
    df = pd.read_csv(csv_path)

    # Temizle ve özellik üret
    df = clean_data(df)
    df = generate_features(df)

    # Feature'lar
    features = ['OverdueDays', 'EksikOdemeOrani', 'KalanOran', 'OdenmediMi', 'InstallmentCount', 'OrtalamaOdeme']
    X = df[features].fillna(0)

    # Modeli yükle
    model = joblib.load(model_path)

    # Tahmin yap
    predictions = model.predict(X)
    df['TahminEdilenRiskSkoru'] = predictions
    df['TahminEdilenRiskSkoru'] = df['TahminEdilenRiskSkoru'].clip(0, 100)

    return df[['ProjectId', 'TahminEdilenRiskSkoru']]

# Test amaçlı çalıştırmak istersen
if __name__ == "__main__":
    result = predict_risk()
    print(result.head())