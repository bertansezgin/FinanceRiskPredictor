"""
Konfigürasyon Yönetimi
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class Config:
    """Ana konfigürasyon sınıfı"""
    
    # Proje kök dizini
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Veri yolları
    DATA_DIR = PROJECT_ROOT / "data"
    MAIN_DATA_FILE = DATA_DIR / "birlesik_risk_verisi.csv"
    NEW_CUSTOMER_FILE = DATA_DIR / "yeni_musteri.csv"
    
    # Model yolları
    MODELS_DIR = PROJECT_ROOT / "models"
    AUTOML_DIR = MODELS_DIR / "automl"
    LINEAR_MODEL_FILE = MODELS_DIR / "linear_model.pkl"
    
    # Çıktı yolları
    REPORTS_DIR = PROJECT_ROOT / "reports"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    
    # Risk hesaplama parametreleri
    RISK_WEIGHTS = {
        'overdue_days': 1.2,
        'payment_missing_ratio': 50.0,
        'remaining_ratio': 40.0,
        'not_paid': 30.0
    }
    
    # Risk kategorileri
    RISK_CATEGORIES = {
        'thresholds': [0, 25, 50, 75, 100],
        'labels': ['Yüksek Risk', 'Orta Risk', 'Düşük Risk', 'Çok Düşük Risk']
    }
    
    # Model parametreleri
    MODEL_CONFIG = {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
    
        'n_jobs': -1
    }
    
    # Risk hesaplama metodları
    RISK_CALCULATION_CONFIG = {
        'method': 'historical_performance',  # Sadece historical_performance - diğerleri data leakage riski nedeniyle kaldırıldı
        'target_months': 6,                  # Risk değerlendirme periyodu
        'explanation': {
            'historical_performance': 'Gerçek payment data tabanlı - SIFIR leakage riski, hiç input feature kullanmıyor'
        }
    }
    
    # Feature engineering parametreleri
    FEATURE_CONFIG = {
        'polynomial_degree': 2,
        'max_features': 50,
        'scaling_method': 'standard',  # 'standard', 'robust', 'minmax'
        'feature_selection_method': 'mutual_info'  # 'mutual_info', 'f_regression'
    }
    
    # Logging konfigürasyonu
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': PROJECT_ROOT / 'logs' / 'finance_risk.log'
    }
    
    # Streamlit konfigürasyonu
    STREAMLIT_CONFIG = {
        'page_title': 'Finansal Risk Tahmin Sistemi',
        'page_icon': '📊',
        'layout': 'wide'
    }
    
    # Sistem sütunları (feature engineering'de hariç tutulacak)
    SYSTEM_COLUMNS = [
        # Sistem/ID sütunları
        'ProjectId', 'ProposalId', 'BranchId', 'AccountNumber',
        'TranDate', 'MaturityDate', 'PaymentDate',
        'UpdateSystemDate', 'CreateSystemDate',
        'UpdateUserName', 'CreateUserName',
        'UpdateHostName', 'CreateHostName',
        'Guid', 'UserName', 'HostName', 'Status',
        
        # DATA LEAKAGE ÖNLEMELERİ
        
        # 1. Target ile doğrudan ilişkili (türetilmiş risk skorları)
        'TemerrütRiskSkoru',        # Target'ın formülünün benzeri
        'Total_Risk_Score',         # Target'ın türevi
        'RiskValue',                # Önceden hesaplanmış risk
        'OdemeGucuSkoru',           # AmountTL kullanıyor
        'Amount_per_Installment',   # AmountTL kullanıyor
        
        # 2. Tahsilat SONRASI bilgiler (Future Information)
        'CollectionAskFER',         # Tahsilat sonrası döviz kuru
        'CollectionBidFER',         # Tahsilat sonrası alış kuru
        'CollectionExchangeFec',    # Tahsilat döviz tipi
        'CollectionStatus',         # Tahsilat durumu
        'ProjectCollectionId',      # Tahsilat ID
        'ProjectCollectionRuleId',  # Tahsilat kural ID
        'ProjectCollectionBankId',  # Tahsilat banka ID
        'ProjectCollectionId_Bank', # Tahsilat banka ID
        'AccountNumber_Collection', # Tahsilat hesap no
        'AccountSuffix_Collection', # Tahsilat hesap eki
        'Amount_Collection',        # Tahsilat tutarı
        'AmountTL',                # Tahsil edilen TL tutarı
        'AmountFEC',               # Tahsil edilen döviz tutarı
        'PaymentAmount',           # Ödeme tutarı
        'DiscountAmount',          # İndirim tutarı
        'CollectionSource',        # Tahsilat kaynağı
        'CollectionType',          # Tahsilat tipi
        'TranBranchId_Collection', # Tahsilat şube
        'ChannelId_Collection',    # Tahsilat kanal
        'UserName_Collection',     # Tahsilat kullanıcı
        'HostName_Collection',     # Tahsilat host
        'SystemDate_Collection',   # Tahsilat sistem tarihi
        'UpdateUserName_Collection',
        'UpdateHostName_Collection',
        'UpdateSystemDate_Collection',
        'HostIP_Collection',
        
        # 3. Karmaşık türetilmiş özellikler (fazlalık)
        'OverdueDays_squared',
        'OverdueDays_cubed',
        'EksikOdemeOrani_squared',
        'OverdueDays_x_EksikOdeme',
        'OdenmediMi_x_OverdueDays',
        
        # 4. Diğer gereksiz/belirsiz sütunlar
        'AccrualFER',
        'AccruedExcDiffBITTAmount',
        'AccruedExcDiffRUSFAmount',
        'SurplusProfitAmount',
        'IncentiveProfitSupportAmount',
    ] + []  # LEAKAGE_COLUMNS will be added below
    
    # SAFE feature'lar - Sadece kredi başlangıcında bilinen
    SAFE_FEATURES = [
        # Kredi başvuru bilgileri
        'ProjectDate', 'InstallmentCount', 'PrincipalAmount', 
        'FundingAmount', 'MonthlyProfitRate',
        
        # Kategorik bilgiler
        'BranchId', 'ProductCode', 'PortfolioClass', 
        'PersonType', 'PaymentType', 'AgreementType',
        
        # Türetilmiş güvenli özellikler
        'TaksitBasinaAnapara', 'FonlamaOrani', 'KrediAyi', 
        'KrediCeyregi', 'KrediTutarKategorisi', 'TaksitSayisiKategorisi',
        'KrediYili', 'AySonuKredi', 'HaftaSonuKredi', 'YazKredisi', 'KisKredisi',
        'IlkOdemeAyi', 'IlkOdemeAySonu', 'BaslangicIlkTaksitGun',
        'TahminiAylikOdeme', 'FaizOraniKategorisi', 'BranchCategory',
        
        # Log transformed features (safe)
        'PrincipalAmount_log', 'FundingAmount_log', 'TahminiAylikOdeme_log',
        
        # Interaction features (safe)
        'KrediTutar_Taksit_Interaksiyon', 'Faiz_Vade_Etkisi', 'OdemeYuku_Oran',
        'VadeRiskSkoru', 'KrediRiskSkoru', 'PrincipalAmount_sqrt', 'InstallmentCount_square',
        
        # Ek güvenli özellikler
        'IsMortgage', 'CollateralType', 'GoodsOrServiceType',
        'DebtFECType', 'MortgageType', 'CampaignDetailId', 'FranchiserId',
    ]

    # LEAKAGE sütunları - ASLA kullanma!
    LEAKAGE_COLUMNS = [
        'AmountTL', 'TranDate', 'RemainingPrincipalAmount',
        'PaymentAmount', 'CollectionStatus', 'ProjectCollectionId',
        'ProjectCollectionRuleId', 'Amount_Collection', 'AmountFEC',
        'DiscountAmount', 'PaymentDate_Bank', 'TransactionType',
        'CollectionAskFER', 'CollectionBidFER', 'CollectionExchangeFec',
        'ProjectCollectionBankId', 'ProjectCollectionId_Bank', 
        'AccountNumber_Collection', 'AccountSuffix_Collection',
        'CollectionSource', 'CollectionType', 'TranBranchId_Collection',
        'ChannelId_Collection', 'UserName_Collection', 'HostName_Collection',
        'SystemDate_Collection', 'UpdateUserName_Collection',
        'UpdateHostName_Collection', 'UpdateSystemDate_Collection',
        'HostIP_Collection', 'MaturityDate', 'PaymentDate',
        # SON 2 EKSİK SÜTUN
        'Amount',      # Target hesaplamada kullanılıyor
        'ProjectId',   # Target hesaplamada kullanılıyor
    ]
    
    # Temel feature'lar (backward compatibility)
    BASE_FEATURES = SAFE_FEATURES.copy()
    
    @classmethod
    def get_system_columns_with_leakage(cls):
        """SYSTEM_COLUMNS + LEAKAGE_COLUMNS birleşimi döndür"""
        return cls.SYSTEM_COLUMNS + cls.LEAKAGE_COLUMNS
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Veri dosyası yolunu döndür"""
        return cls.DATA_DIR / filename
    
    @classmethod
    def get_model_path(cls, filename: str) -> Path:
        """Model dosyası yolunu döndür"""
        return cls.MODELS_DIR / filename
    
    @classmethod
    def get_automl_path(cls, filename: str) -> Path:
        """AutoML model dosyası yolunu döndür"""
        return cls.AUTOML_DIR / filename
    
    @classmethod
    def get_report_path(cls, filename: str) -> Path:
        """Rapor dosyası yolunu döndür"""
        return cls.REPORTS_DIR / filename
    
    @classmethod
    def get_plot_path(cls, filename: str) -> Path:
        """Görselleştirme dosyası yolunu döndür"""
        return cls.PLOTS_DIR / filename
    
    @classmethod
    def create_directories(cls):
        """Gerekli dizinleri oluştur"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.AUTOML_DIR,
            cls.REPORTS_DIR,
            cls.PLOTS_DIR,
            cls.LOGGING_CONFIG['file'].parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Dizinler oluşturuldu")
    
    @classmethod
    def setup_logging(cls):
        """Logging'i konfigüre et"""
        cls.create_directories()  # Log dizinini oluştur
        
        logging.basicConfig(
            level=getattr(logging, cls.LOGGING_CONFIG['level']),
            format=cls.LOGGING_CONFIG['format'],
            handlers=[
                logging.FileHandler(cls.LOGGING_CONFIG['file']),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Logging konfigüre edildi")
    






# Global config instance
config = Config()

# Initialize logging on import
try:
    config.setup_logging()
except Exception as e:
    print(f"Logging kurulum hatası: {e}")
