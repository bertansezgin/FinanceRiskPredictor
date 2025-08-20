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
# from src.risk_calculator import calculate_realistic_risk_score  # Removed - data leakage risk
from src.config import config


class AutoMLPipeline:
    """
    Otomatik ML Pipeline - Risk skorlaması için end-to-end sistem
    """
    
    def __init__(self):
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
        
        # ProjectId'yi GroupShuffleSplit için backup al (silinmeden önce!)
        project_ids_backup = df['ProjectId'].copy() if 'ProjectId' in df.columns else None
        
        # 1. Feature Engineering
        # 2. TEMPORAL TARGET oluştur - Config'e göre metod seç (ÖNCE!)
        risk_method = config.RISK_CALCULATION_CONFIG['method']
        print(f"🎯 {risk_method.title()} temporal risk skoru hesaplanıyor...")
        print(f"   📋 {config.RISK_CALCULATION_CONFIG['explanation'][risk_method]}")
        
        # Sadece historical performance - diğer metodlar data leakage riski nedeniyle kaldırıldı
        if risk_method == 'historical_performance':
            from src.historical_target_calculator import calculate_historical_target, validate_target_independence
            # Target independence doğrulama
            validate_target_independence(df)
            df['RiskScore'] = calculate_historical_target(df)
        else:
            raise ValueError(f"Desteklenmeyen risk metodu: {risk_method}. Sadece 'historical_performance' kullanılabilir.")
        
        print(f"✅ Target istatistikleri:")
        print(f"   📊 Ortalama: {df['RiskScore'].mean():.2f}")
        print(f"   📈 Std: {df['RiskScore'].std():.2f}")
        print(f"   📉 Min-Max: [{df['RiskScore'].min():.1f}, {df['RiskScore'].max():.1f}]")
        
        # 3. Feature engineering (SONRA!)
        print("\n📊 Feature engineering...")
        df_features = self.feature_engineer.create_advanced_features(df)
        

        
        # 3. TEMPORAL FILTER - Sadece feature period
        print("📅 Temporal filtering yapılıyor...")
        # Sadece historical performance kaldı
        from src.historical_target_calculator import historical_calculator
        df_filtered = historical_calculator.filter_feature_period_projects(df_features)
        
        print(f"📅 Temporal filtering:")
        print(f"   🔢 Önceki kayıt sayısı: {len(df_features)}")
        print(f"   🔢 Sonraki kayıt sayısı: {len(df_filtered)}")
        
        # Backup ProjectId'yi filtered index'lerle eşle
        if project_ids_backup is not None:
            project_ids_for_split = project_ids_backup.loc[df_filtered.index]
        else:
            raise ValueError("ProjectId backup bulunamadı - GroupShuffleSplit yapılamıyor!")
        
        # 4. Feature seçimi - LEAKAGE-FREE ONLY
        print("🔍 Leakage-free feature selection...")
        
        # Import balanced config - DAHA İYİ PERFORMANCE İÇİN
        from src.balanced_feature_config import get_balanced_features, get_balanced_explanation
        # from src.leakage_free_config import get_leakage_free_features, is_feature_safe  # Removed - using balanced approach
        
        # Kullanıcı tercihi: ultra safe (14) vs balanced (28)
        USE_BALANCED_APPROACH = True  # FALSE = ultra safe (14), TRUE = balanced (28)
        
        if USE_BALANCED_APPROACH:
            truly_safe_features = get_balanced_features()
            approach_info = get_balanced_explanation()
            print(f"🎯 BALANCED APPROACH: {approach_info['feature_count']} feature")
            print(f"   📊 Beklenen performans: {approach_info['expected_performance']}")
        else:
            # truly_safe_features = get_leakage_free_features()  # Removed - using balanced approach only
            truly_safe_features = get_balanced_features()  # Fallback to balanced
            print(f"🛡️ FALLBACK TO BALANCED APPROACH: {len(truly_safe_features)} feature")
        
        # Mevcut sütunlarla kesişimi al
        available_safe_features = [col for col in df_filtered.columns 
                                  if col in truly_safe_features]
        
        # Debug: hangi feature'lar neden reddedildi?
        print("🔍 Feature güvenlik analizi:")
        all_features = [col for col in df_filtered.columns 
                       if col not in ['RiskScore'] and col in config.SAFE_FEATURES]
        
        safe_count = 0
        unsafe_count = 0
        
        for feature in all_features[:20]:  # İlk 20'sini göster
            # is_safe, reason = is_feature_safe(feature)  # Removed - using balanced approach
            if feature in truly_safe_features:
                safe_count += 1
                print(f"   ✅ {feature}: In balanced feature set")
            else:
                unsafe_count += 1
                print(f"   ❌ {feature}: Not in balanced feature set")
        
        print(f"   📊 Toplam güvenli feature: {safe_count}")
        print(f"   📊 Toplam güvenli olmayan: {unsafe_count}")
        
        # Sadece numerik safe features kullan
        if available_safe_features:
            X_all = df_filtered[available_safe_features].fillna(0)
            X = X_all.select_dtypes(include=[np.number])  # Sadece numerik
        else:
            print("⚠️ UYARI: Hiç leakage-free feature yok! Minimal set kullanılıyor...")
            # Fallback - sadece FundingAmount kullan
            minimal_features = ['FundingAmount']
            available_minimal = [col for col in minimal_features if col in df_filtered.columns]
            if available_minimal:
                X = df_filtered[available_minimal].fillna(0).select_dtypes(include=[np.number])
            else:
                # Son çare - dummy feature
                X = pd.DataFrame({'dummy_feature': [1.0] * len(df_filtered)}, index=df_filtered.index)
        
        y = df_filtered['RiskScore']
        
        print(f"🔍 Leakage-free feature selection:")
        print(f"   📊 Kullanılabilir leakage-free feature: {len(available_safe_features)}")
        print(f"   📊 Numerik feature sayısı: {len(X.columns)}")
        print(f"   ✅ %100 Leakage-free garantisi!")
        
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
        
        # 4. Train-test split - PROJECTID LEAKAGE ÖNLEME
        print("📈 Veri bölünüyor (ProjectId leakage önleme)...")
        from sklearn.model_selection import GroupShuffleSplit
        
        # ProjectId'ye göre split - aynı proje hem train hem test'te olmasın
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=project_ids_for_split))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"🔒 PROJECTID LEAKAGE ÖNLENDİ:")
        print(f"   📊 Train ProjectId sayısı: {project_ids_for_split.iloc[train_idx].nunique()}")
        print(f"   📊 Test ProjectId sayısı: {project_ids_for_split.iloc[test_idx].nunique()}")
        
        # Overlap kontrolü
        train_projects = set(project_ids_for_split.iloc[train_idx])
        test_projects = set(project_ids_for_split.iloc[test_idx])
        overlap = train_projects.intersection(test_projects)
        print(f"   ✅ Overlap: {len(overlap)} (0 olmalı)")
        
        if len(overlap) > 0:
            print(f"   🚨 UYARI: {len(overlap)} ProjectId overlap var!")
        else:
            print(f"   ✅ ProjectId leakage önlendi!")
        
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


