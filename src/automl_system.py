"""
AutoML Sistemi - Risk Skorlaması için ML Pipeline
"""

import numpy as np
import pandas as pd
import os
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import AdvancedFeatureEngineering
from src.advanced_models import AdvancedRiskModels
from src.risk_calculator import calculate_realistic_risk_score
from src.config import config


class AutoMLPipeline:
    """
    Otomatik ML Pipeline - Risk skorlaması için end-to-end sistem
    """
    
    def __init__(self, optimize_hyperparams=False, n_trials=30):
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.feature_engineer = AdvancedFeatureEngineering()
        # Orijinal AdvancedRiskModels sınıfını kullan (duplicate değil)
        self.models = AdvancedRiskModels()
        self.scaler = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        
    def run_automl(self, df):
        """
        Ana AutoML akışı - tam otomatik risk skorlaması
        
        Args:
            df: Ham veri DataFrame
            
        Returns:
            dict: Sonuçlar ve metrikler
        """
        print("🚀 AutoML Pipeline başlıyor...")
        
        # 1. Feature Engineering
        print("\n📊 Feature engineering...")
        df_features = self.feature_engineer.create_advanced_features(df)
        
        # 2. TEMPORAL TARGET oluştur - Config'e göre metod seç
        risk_method = config.RISK_CALCULATION_CONFIG['method']
        print(f"🎯 {risk_method.title()} temporal risk skoru hesaplanıyor...")
        print(f"   📋 {config.RISK_CALCULATION_CONFIG['explanation'][risk_method]}")
        
        if risk_method == 'deterministic':
            from src.risk_calculator import calculate_deterministic_risk_score
            df_features['RiskScore'] = calculate_deterministic_risk_score(df_features)
        else:  # stochastic
            from src.risk_calculator import calculate_temporal_risk_score
            df_features['RiskScore'] = calculate_temporal_risk_score(df_features)
        
        print(f"✅ Target istatistikleri:")
        print(f"   📊 Ortalama: {df_features['RiskScore'].mean():.2f}")
        print(f"   📈 Std: {df_features['RiskScore'].std():.2f}")
        print(f"   📉 Min-Max: [{df_features['RiskScore'].min():.1f}, {df_features['RiskScore'].max():.1f}]")
        
        # 3. TEMPORAL FILTER - Sadece feature period
        print("📅 Temporal filtering yapılıyor...")
        if risk_method == 'deterministic':
            from src.risk_calculator import deterministic_calculator
            df_filtered = deterministic_calculator.filter_feature_period_projects(df_features)
        else:  # stochastic
            from src.risk_calculator import temporal_calculator
            df_filtered = temporal_calculator.filter_feature_period_projects(df_features)
        
        print(f"📅 Temporal filtering:")
        print(f"   🔢 Önceki kayıt sayısı: {len(df_features)}")
        print(f"   🔢 Sonraki kayıt sayısı: {len(df_filtered)}")
        
        # 4. Feature seçimi - SAFE ONLY
        print("🔍 Safe feature selection...")
        safe_feature_cols = [col for col in df_filtered.columns 
                            if col in config.SAFE_FEATURES and col in df_filtered.columns]
        
        # Sadece numerik safe features kullan (correlation için)
        X_all = df_filtered[safe_feature_cols].fillna(0)
        X = X_all.select_dtypes(include=[np.number])  # Sadece numerik
        y = df_filtered['RiskScore']
        
        print(f"🔍 Feature selection:")
        print(f"   📊 Safe feature sayısı: {len(safe_feature_cols)}")
        print(f"   📊 Numerik feature sayısı: {len(X.columns)}")
        print(f"   ✅ Safe features only!")
        
        # 5. Data leakage kontrolü
        print("\n🔍 Data leakage kontrolü...")
        
        correlation_check = X.corrwith(y).abs().sort_values(ascending=False)
        high_corr_features = correlation_check[correlation_check > 0.8]
        
        if len(high_corr_features) > 0:
            print(f"⚠️ UYARI: {len(high_corr_features)} feature yüksek korelasyon!")
            for feature, corr in high_corr_features.items():
                print(f"   🚨 {feature}: {corr:.3f}")
        else:
            print("✅ Data leakage kontrolü BAŞARILI!")
        
        # 4. Train-test split
        print("📈 Veri bölünüyor...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 5. Scaling
        print("⚖️ Ölçeklendirme...")
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 6. Model eğitimi
        print("🤖 Modeller eğitiliyor...")
        results_df = self.models.train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 7. En iyi modeli bul
        self.best_model_name, self.best_model = self.models.get_best_model()
        print(f"\n🏆 En iyi model: {self.best_model_name}")
        
        # 8. Feature names kaydet
        self.feature_names = X_train.columns.tolist()
        
        # 9. Modeli kaydet
        self._save_model_artifacts()
        
        # 10. Sonuçları hazırla
        metrics = self._calculate_metrics(X_test_scaled, y_test)
        
        return {
            'best_model_name': self.best_model_name,
            'best_model': self.best_model,
            'results_df': results_df,
            'metrics': metrics,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
    
    def _save_model_artifacts(self):
        """Model artefaktlarını kaydet"""
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Dosya yolları
        model_path = f"models/automl/best_model_{self.best_model_name}_{timestamp}.pkl"
        scaler_path = f"models/automl/scaler_{timestamp}.pkl"
        features_path = f"models/automl/features_{timestamp}.pkl"
        info_path = f"models/automl/model_info_{timestamp}.json"
        
        # Kaydet
        os.makedirs("models/automl", exist_ok=True)
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, features_path)
        
        # Model bilgileri
        model_info = {
            'timestamp': timestamp,
            'best_model_name': self.best_model_name,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path,
            'feature_count': len(self.feature_names),
            'hyperparams_optimized': self.optimize_hyperparams,
            'n_trials': self.n_trials
        }
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
            
        print(f"✅ Model kaydedildi: {model_path}")
        
    def _calculate_metrics(self, X_test, y_test):
        """Test metriklerini hesapla"""
        
        y_pred = self.best_model.predict(X_test)
        
        return {
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'model_name': self.best_model_name
        }


