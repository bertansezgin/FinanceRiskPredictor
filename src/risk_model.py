import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

from src.risk_calculator import calculate_risk_from_dataframe
from src.config import config

logger = logging.getLogger(__name__)

def train_and_save_model(df, model_path=None):
    """
    LinearRegression modelini eğitir ve disk'e kaydeder.
    """
    try:
        if model_path is None:
            model_path = config.LINEAR_MODEL_FILE
        
        features = config.BASE_FEATURES
        
        # Gerekli sütunları kontrol et
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Eksik özellikler: {missing_features}")

        X = df[features].fillna(0)

        # Merkezi risk hesaplama kullan
        y = calculate_risk_from_dataframe(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.MODEL_CONFIG['test_size'], 
            random_state=config.MODEL_CONFIG['random_state']
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model eğitimi tamamlandı - MSE: {mse:.4f}, R2: {r2:.4f}")

        # Dizin oluştur
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Kaydet
        joblib.dump(model, model_path)
        logger.info(f"Model kaydedildi: {model_path}")
        
        return model, {'mse': mse, 'r2': r2}
        
    except Exception as e:
        logger.error(f"Model eğitimi hatası: {e}")
        raise