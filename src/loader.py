"""
Veri Yükleme - TEMPORAL SPLIT READY (Data Leakage Yok)
"""

import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def load_data(path: str, validate: bool = True) -> pd.DataFrame:
    """
    CSV dosyasını okur - TEMPORAL SPLIT HAZIR

    NOT: Feature engineering burada yapılmaz!
    Sadece basic cleaning, feature engineering AdvancedFeatureEngineering'de

    Args:
        path: CSV dosya yolu
        validate: Temel doğrulama yapılsın mı

    Returns:
        Raw DataFrame (sadece basic cleaning)
    """
    try:
        # Dosya kontrolü
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")

        # CSV okuma
        df = pd.read_csv(path)
        logger.info(f"Raw veri yüklendi: {len(df)} satır, {len(df.columns)} sütun - {path}")

        if df.empty:
            raise ValueError("Yüklenen veri boş")

        # SADECE BASIC CLEANING - FEATURE ENGINEERING YOK!
        if validate:
            if df.shape[0] < 10:
                logger.warning("Çok az veri var (< 10 satır)")

            # Sadece temel temizlik
            df = _basic_clean_only(df)
            logger.info(f"Temiz veri: {len(df)} satır, {len(df.columns)} sütun")

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


def _basic_clean_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sadece temel temizlik - FEATURE ENGINEERING YOK!

    PRENSIP: Raw data'yı bozmadan, sadece gerekli temizlik
    """

    df_clean = df.copy()

    print("🧹 Basic cleaning başlıyor...")

    # 1. Tamamen boş sütunları sil
    empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
    if empty_cols:
        df_clean = df_clean.drop(columns=empty_cols)
        print(f"   ❌ {len(empty_cols)} boş sütun silindi")

    # 2. Sadece sistem/log sütunlarını temizle (data değil!)
    system_log_keywords = [
        'Guid', 'SystemDate', 'UserName', 'HostName',
        'UpdateUser', 'CreateUser', 'UpdateHost', 'CreateHost',
        'HostIP', 'UpdateSystemDate', 'CreateSystemDate'
    ]

    system_cols = []
    for col in df_clean.columns:
        if any(keyword in col for keyword in system_log_keywords):
            system_cols.append(col)

    if system_cols:
        df_clean = df_clean.drop(columns=system_cols, errors='ignore')
        print(f"   🗑️ {len(system_cols)} sistem sütunu silindi")

    # 3. Tarih sütunlarını düzelt (basic parsing only)
    date_columns = ['ProjectDate', 'MaturityDate', 'TranDate', 'FirstInstallmentDate']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    # 4. Temel sayısal sütunları düzelt
    numeric_columns = ['PrincipalAmount', 'FundingAmount', 'InstallmentCount', 'MonthlyProfitRate']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    print(f"✅ Basic cleaning tamamlandı: {len(df_clean)} satır, {len(df_clean.columns)} sütun")

    # ÖNEMLI: Hiçbir derived feature oluşturma!
    # - KalanOran oluşturma
    # - OverdueDays hesaplama
    # - TahsilatYapilmadi kontrol etme
    # Bunlar AdvancedFeatureEngineering'de yapılacak

    return df_clean


