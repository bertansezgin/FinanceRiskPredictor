from src.predict import predict_risk
from src.loader import load_data
from src.preprocessing import clean_data, generate_features
from src.risk_model import train_and_save_model
from src.config import config
import logging

# Logging'i başlat
config.setup_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        logger.info("Ana program başlatılıyor")
        
        path = config.MAIN_DATA_FILE
        df = load_data(path)
        df = clean_data(df)
        df = generate_features(df)

        model, metrics = train_and_save_model(df)
        logger.info(f"Model eğitimi tamamlandı: {metrics}")

        result = predict_risk()
        print(result.head())
        
        logger.info("Ana program başarıyla tamamlandı")
        
    except Exception as e:
        logger.error(f"Ana program hatası: {e}")
        raise