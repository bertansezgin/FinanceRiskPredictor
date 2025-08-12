"""
Veri Doğrulama Modülü
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from src.config import ValidationConfig

logger = logging.getLogger(__name__)


class DataValidator:
    """Veri doğrulama sınıfı"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
    
    def validate_dataframe(self, df: pd.DataFrame, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        DataFrame'i doğrula
        
        Args:
            df: Doğrulanacak DataFrame
            strict: Katı mod (tüm kontroller)
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Boş DataFrame kontrolü
            if df.empty:
                errors.append("DataFrame boş")
                return False, errors
            
            # Gerekli sütunlar kontrolü
            missing_cols = self._check_required_columns(df)
            if missing_cols:
                errors.extend(missing_cols)
            
            # Veri tipi kontrolü
            if strict:
                type_errors = self._check_column_types(df)
                if type_errors:
                    errors.extend(type_errors)
            
            # Değer aralığı kontrolü
            range_errors = self._check_value_ranges(df)
            if range_errors:
                errors.extend(range_errors)
            
            # Eksik değer kontrolü
            missing_errors = self._check_missing_values(df)
            if missing_errors:
                errors.extend(missing_errors)
            
            # Duplicate kontrolü
            if df.duplicated().any():
                duplicate_count = df.duplicated().sum()
                errors.append(f"Duplicate satır sayısı: {duplicate_count}")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info("DataFrame doğrulama başarılı")
            else:
                logger.warning(f"DataFrame doğrulama başarısız: {len(errors)} hata")
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"DataFrame doğrulama hatası: {e}")
            return False, [f"Doğrulama hatası: {str(e)}"]
    
    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Gerekli sütunları kontrol et"""
        errors = []
        missing_cols = [col for col in self.config.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_cols:
            errors.append(f"Eksik gerekli sütunlar: {missing_cols}")
        
        return errors
    
    def _check_column_types(self, df: pd.DataFrame) -> List[str]:
        """Sütun veri tiplerini kontrol et"""
        errors = []
        
        for col, expected_type in self.config.COLUMN_TYPES.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    try:
                        # Tip dönüşümü dene
                        df[col].astype(expected_type)
                    except (ValueError, TypeError):
                        errors.append(f"Sütun '{col}' tip hatası: beklenen {expected_type}, mevcut {actual_type}")
        
        return errors
    
    def _check_value_ranges(self, df: pd.DataFrame) -> List[str]:
        """Değer aralıklarını kontrol et"""
        errors = []
        
        for col, (min_val, max_val) in self.config.VALUE_RANGES.items():
            if col in df.columns:
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    actual_min = col_data.min()
                    actual_max = col_data.max()
                    
                    if actual_min < min_val:
                        errors.append(f"Sütun '{col}' minimum değer hatası: {actual_min} < {min_val}")
                    
                    if max_val != float('inf') and actual_max > max_val:
                        errors.append(f"Sütun '{col}' maksimum değer hatası: {actual_max} > {max_val}")
        
        return errors
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[str]:
        """Eksik değerleri kontrol et"""
        errors = []
        
        for col in df.columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            
            if missing_ratio > self.config.MAX_MISSING_RATIO:
                errors.append(f"Sütun '{col}' çok fazla eksik değer: %{missing_ratio*100:.1f}")
        
        return errors
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'i temizle ve düzelt
        
        Args:
            df: Temizlenecek DataFrame
            
        Returns:
            Temizlenmiş DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Duplicate'leri kaldır
            if df_clean.duplicated().any():
                df_clean = df_clean.drop_duplicates()
                logger.info("Duplicate satırlar kaldırıldı")
            
            # Veri tipi düzeltmeleri
            for col, expected_type in self.config.COLUMN_TYPES.items():
                if col in df_clean.columns:
                    try:
                        df_clean[col] = df_clean[col].astype(expected_type)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Sütun '{col}' tip dönüşümü başarısız: {e}")
            
            # Değer aralığı düzeltmeleri
            for col, (min_val, max_val) in self.config.VALUE_RANGES.items():
                if col in df_clean.columns:
                    if max_val != float('inf'):
                        df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
                    else:
                        df_clean[col] = df_clean[col].clip(lower=min_val)
            
            logger.info("DataFrame temizleme tamamlandı")
            return df_clean
            
        except Exception as e:
            logger.error(f"DataFrame temizleme hatası: {e}")
            raise
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Veri kalitesi raporu oluştur
        
        Args:
            df: Analiz edilecek DataFrame
            
        Returns:
            Veri kalitesi raporu
        """
        try:
            report = {
                'basic_info': {
                    'shape': df.shape,
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'dtypes': df.dtypes.value_counts().to_dict()
                },
                'missing_values': {
                    'total_missing': df.isnull().sum().sum(),
                    'missing_by_column': df.isnull().sum().to_dict(),
                    'missing_ratio_by_column': (df.isnull().sum() / len(df)).to_dict()
                },
                'duplicates': {
                    'duplicate_count': df.duplicated().sum(),
                    'duplicate_ratio': df.duplicated().sum() / len(df)
                },
                'numeric_stats': {},
                'categorical_stats': {}
            }
            
            # Sayısal sütun istatistikleri
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                report['numeric_stats'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'zero_count': (df[col] == 0).sum(),
                    'negative_count': (df[col] < 0).sum()
                }
            
            # Kategorik sütun istatistikleri
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                report['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'value_counts': df[col].value_counts().head(10).to_dict()
                }
            
            logger.info("Veri kalitesi raporu oluşturuldu")
            return report
            
        except Exception as e:
            logger.error(f"Veri kalitesi raporu hatası: {e}")
            return {}


def validate_file_path(file_path: str) -> bool:
    """
    Dosya yolunu doğrula
    
    Args:
        file_path: Dosya yolu
        
    Returns:
        Dosya geçerli mi
    """
    import os
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Dosya bulunamadı: {file_path}")
            return False
        
        if not os.path.isfile(file_path):
            logger.error(f"Geçersiz dosya: {file_path}")
            return False
        
        if os.path.getsize(file_path) == 0:
            logger.error(f"Dosya boş: {file_path}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Dosya doğrulama hatası: {e}")
        return False


def validate_model_inputs(
    overdue_days: float,
    payment_missing_ratio: float,
    remaining_ratio: float,
    not_paid: int
) -> Tuple[bool, List[str]]:
    """
    Model girdi değerlerini doğrula
    
    Args:
        overdue_days: Gecikme günleri
        payment_missing_ratio: Eksik ödeme oranı
        remaining_ratio: Kalan borç oranı
        not_paid: Ödenmedi mi
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Gecikme günleri kontrolü
        if not isinstance(overdue_days, (int, float)):
            errors.append("Gecikme günleri sayısal değer olmalı")
        elif overdue_days < 0:
            errors.append("Gecikme günleri negatif olamaz")
        elif overdue_days > 3650:  # 10 yıl
            errors.append("Gecikme günleri çok yüksek (>10 yıl)")
        
        # Ödeme oranı kontrolü
        if not isinstance(payment_missing_ratio, (int, float)):
            errors.append("Eksik ödeme oranı sayısal değer olmalı")
        elif payment_missing_ratio < 0 or payment_missing_ratio > 1:
            errors.append("Eksik ödeme oranı 0-1 aralığında olmalı")
        
        # Kalan oran kontrolü
        if not isinstance(remaining_ratio, (int, float)):
            errors.append("Kalan borç oranı sayısal değer olmalı")
        elif remaining_ratio < 0 or remaining_ratio > 1:
            errors.append("Kalan borç oranı 0-1 aralığında olmalı")
        
        # Ödenmedi mi kontrolü
        if not isinstance(not_paid, (int, bool)):
            errors.append("Ödenmedi mi değeri 0/1 veya True/False olmalı")
        elif not_paid not in [0, 1, True, False]:
            errors.append("Ödenmedi mi değeri 0, 1, True veya False olmalı")
        
        is_valid = len(errors) == 0
        return is_valid, errors
        
    except Exception as e:
        logger.error(f"Model girdi doğrulama hatası: {e}")
        return False, [f"Doğrulama hatası: {str(e)}"]


# Global validator instance
default_validator = DataValidator()

# Convenience functions
def validate_dataframe(*args, **kwargs):
    """Varsayılan validator ile DataFrame doğrula"""
    return default_validator.validate_dataframe(*args, **kwargs)

def clean_dataframe(*args, **kwargs):
    """Varsayılan validator ile DataFrame temizle"""
    return default_validator.clean_dataframe(*args, **kwargs)

def get_data_quality_report(*args, **kwargs):
    """Varsayılan validator ile veri kalitesi raporu oluştur"""
    return default_validator.get_data_quality_report(*args, **kwargs)
