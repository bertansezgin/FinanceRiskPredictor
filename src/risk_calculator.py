"""
Merkezi Risk Hesaplama Modülü
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RiskWeights:
    """Risk hesaplama katsayıları"""
    OVERDUE_DAYS = 1.2
    PAYMENT_MISSING_RATIO = 50.0
    REMAINING_RATIO = 40.0
    NOT_PAID = 30.0
    
    @classmethod
    def get_weights_dict(cls) -> Dict[str, float]:
        """Katsayıları dictionary olarak döndür"""
        return {
            'overdue_days': cls.OVERDUE_DAYS,
            'payment_missing_ratio': cls.PAYMENT_MISSING_RATIO,
            'remaining_ratio': cls.REMAINING_RATIO,
            'not_paid': cls.NOT_PAID
        }


class RiskCalculator:
    """Merkezi risk hesaplama sınıfı"""
    
    def __init__(self, weights: RiskWeights = None):
        self.weights = weights or RiskWeights()
        
    def calculate_risk_score(
        self,
        overdue_days: Union[float, pd.Series, np.ndarray],
        payment_missing_ratio: Union[float, pd.Series, np.ndarray],
        remaining_ratio: Union[float, pd.Series, np.ndarray],
        not_paid: Union[float, pd.Series, np.ndarray]
    ) -> Union[float, pd.Series, np.ndarray]:
        """
        Risk skorunu hesapla
        
        Args:
            overdue_days: Gecikme günleri
            payment_missing_ratio: Eksik ödeme oranı (0-1)
            remaining_ratio: Kalan borç oranı (0-1)
            not_paid: Ödenmedi mi (0 veya 1)
            
        Returns:
            Risk skoru (0-100 arası, düşük skor = yüksek risk)
        """
        try:
            risk_score = (
                100 
                - overdue_days * self.weights.OVERDUE_DAYS
                - payment_missing_ratio * self.weights.PAYMENT_MISSING_RATIO
                - remaining_ratio * self.weights.REMAINING_RATIO
                - not_paid * self.weights.NOT_PAID
            )
            
            # 0-100 aralığına sınırla
            if isinstance(risk_score, (pd.Series, np.ndarray)):
                return risk_score.clip(0, 100)
            else:
                return max(0, min(100, risk_score))
                
        except Exception as e:
            logger.error(f"Risk skoru hesaplama hatası: {e}")
            raise ValueError(f"Risk skoru hesaplanamadı: {e}")
    
    def calculate_risk_from_dataframe(
        self,
        df: pd.DataFrame,
        overdue_col: str = 'OverdueDays',
        payment_missing_col: str = 'EksikOdemeOrani',
        remaining_col: str = 'KalanOran',
        not_paid_col: str = 'OdenmediMi'
    ) -> pd.Series:
        """
        DataFrame'den risk skorunu hesapla
        
        Args:
            df: Veri DataFrame'i
            overdue_col: Gecikme günleri sütun adı
            payment_missing_col: Eksik ödeme oranı sütun adı
            remaining_col: Kalan borç oranı sütun adı
            not_paid_col: Ödenmedi mi sütun adı
            
        Returns:
            Risk skorları serisi
        """
        try:
            # Sütunların varlığını kontrol et
            required_cols = [overdue_col, payment_missing_col, remaining_col, not_paid_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Eksik sütunlar: {missing_cols}")
            
            # Eksik değerleri doldur
            df_clean = df.copy()
            df_clean[overdue_col] = df_clean[overdue_col].fillna(0)
            df_clean[payment_missing_col] = df_clean[payment_missing_col].fillna(0)
            df_clean[remaining_col] = df_clean[remaining_col].fillna(0)
            df_clean[not_paid_col] = df_clean[not_paid_col].fillna(0)
            
            return self.calculate_risk_score(
                df_clean[overdue_col],
                df_clean[payment_missing_col],
                df_clean[remaining_col],
                df_clean[not_paid_col]
            )
            
        except Exception as e:
            logger.error(f"DataFrame'den risk hesaplama hatası: {e}")
            raise
    
    def get_risk_category(
        self,
        risk_score: Union[float, pd.Series, np.ndarray]
    ) -> Union[str, pd.Series]:
        """
        Risk skorundan kategori belirle
        
        Args:
            risk_score: Risk skoru (0-100)
            
        Returns:
            Risk kategorisi
        """
        if isinstance(risk_score, (pd.Series, np.ndarray)):
            return pd.cut(
                risk_score,
                bins=[0, 25, 50, 75, 100],
                labels=['Yüksek Risk', 'Orta Risk', 'Düşük Risk', 'Çok Düşük Risk'],
                include_lowest=True
            )
        else:
            if risk_score < 25:
                return 'Yüksek Risk'
            elif risk_score < 50:
                return 'Orta Risk'
            elif risk_score < 75:
                return 'Düşük Risk'
            else:
                return 'Çok Düşük Risk'
    
    def validate_inputs(
        self,
        overdue_days: Union[float, pd.Series, np.ndarray],
        payment_missing_ratio: Union[float, pd.Series, np.ndarray],
        remaining_ratio: Union[float, pd.Series, np.ndarray],
        not_paid: Union[float, pd.Series, np.ndarray]
    ) -> bool:
        """
        Girdi değerlerini doğrula
        
        Args:
            overdue_days: Gecikme günleri
            payment_missing_ratio: Eksik ödeme oranı
            remaining_ratio: Kalan borç oranı
            not_paid: Ödenmedi mi
            
        Returns:
            Doğrulama başarılı mı
        """
        try:
            # Negatif değer kontrolü
            if isinstance(overdue_days, (pd.Series, np.ndarray)):
                if (overdue_days < 0).any():
                    raise ValueError("Gecikme günleri negatif olamaz")
            else:
                if overdue_days < 0:
                    raise ValueError("Gecikme günleri negatif olamaz")
            
            # Oran kontrolü (0-1 arası olmalı)
            for ratio, name in [(payment_missing_ratio, 'Eksik ödeme oranı'), 
                               (remaining_ratio, 'Kalan borç oranı')]:
                if isinstance(ratio, (pd.Series, np.ndarray)):
                    if (ratio < 0).any() or (ratio > 1).any():
                        raise ValueError(f"{name} 0-1 aralığında olmalı")
                else:
                    if ratio < 0 or ratio > 1:
                        raise ValueError(f"{name} 0-1 aralığında olmalı")
            
            # Binary değer kontrolü (0 veya 1)
            if isinstance(not_paid, (pd.Series, np.ndarray)):
                unique_vals = pd.Series(not_paid).unique()
                if not set(unique_vals).issubset({0, 1}):
                    raise ValueError("Ödenmedi mi değeri 0 veya 1 olmalı")
            else:
                if not_paid not in [0, 1]:
                    raise ValueError("Ödenmedi mi değeri 0 veya 1 olmalı")
            
            return True
            
        except Exception as e:
            logger.error(f"Girdi doğrulama hatası: {e}")
            raise


# Global risk calculator instance
default_risk_calculator = RiskCalculator()

# Convenience functions
def calculate_risk_score(*args, **kwargs):
    """Varsayılan risk calculator ile risk skoru hesapla"""
    return default_risk_calculator.calculate_risk_score(*args, **kwargs)

def calculate_risk_from_dataframe(*args, **kwargs):
    """Varsayılan risk calculator ile DataFrame'den risk skoru hesapla"""
    return default_risk_calculator.calculate_risk_from_dataframe(*args, **kwargs)

def get_risk_category(*args, **kwargs):
    """Varsayılan risk calculator ile risk kategorisi belirle"""
    return default_risk_calculator.get_risk_category(*args, **kwargs)
