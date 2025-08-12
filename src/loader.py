import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from src.validation import validate_file_path, validate_dataframe
from src.config import config

logger = logging.getLogger(__name__)

def load_data(path: str, validate: bool = True) -> pd.DataFrame:
    """
    CSV dosyasını okur ve DataFrame olarak döner.
    
    Args:
        path: CSV dosya yolu
        validate: Veri doğrulama yapılsın mı
        
    Returns:
        Yüklenen DataFrame
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        ValueError: Veri doğrulama başarısızsa
    """
    try:
        # Dosya yolu doğrulama
        if not validate_file_path(path):
            raise FileNotFoundError(f"Geçersiz dosya yolu: {path}")
        
        # CSV okuma
        df = pd.read_csv(path)
        logger.info(f"Veri yüklendi: {len(df)} satır, {len(df.columns)} sütun - {path}")
        
        if df.empty:
            raise ValueError("Yüklenen veri boş")
        
        # Veri doğrulama
        if validate:
            is_valid, errors = validate_dataframe(df, strict=False)
            if not is_valid:
                logger.warning(f"Veri doğrulama uyarıları: {errors}")
                # Uyarı ver ama devam et
        
        return df
        
    except FileNotFoundError:
        logger.error(f"Dosya bulunamadı: {path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Dosya boş: {path}")
        raise ValueError(f"Dosya boş: {path}")
    except pd.errors.ParserError as e:
        logger.error(f"CSV parse hatası: {e}")
        raise ValueError(f"CSV okuma hatası: {e}")
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {e}")
        raise
