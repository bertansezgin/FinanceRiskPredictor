"""
GELİŞTİRİLMİŞ Makine Öğrenmesi Modelleri
Data leakage temizlenmiş, optimize edilmiş parametreler
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class AdvancedRiskModels:
    """Sadeleştirilmiş risk tahmin modelleri"""
    
    def __init__(self):
        self.models = {
            # Baseline model - Güçlendirilmiş
            'Ridge': Ridge(alpha=10.0, solver='auto'),
            
            # Optimize edilmiş modeller - Daha iyi parametreler
            'RandomForest': RandomForestRegressor(
                n_estimators=200,          # Artırıldı
                max_depth=15,              # Artırıldı
                min_samples_split=5,       # Azaltıldı
                min_samples_leaf=2,        # Azaltıldı
                max_features=0.8,          # Eklendi
                bootstrap=True,
                oob_score=True,           # Eklendi
                random_state=42,
                n_jobs=-1
            ),
            
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,          # Artırıldı
                learning_rate=0.08,        # Azaltıldı (daha stabil)
                max_depth=6,               # Artırıldı
                min_samples_split=5,       # Azaltıldı
                min_samples_leaf=3,        # Azaltıldı
                subsample=0.9,             # Eklendi (regularization)
                max_features=0.8,          # Eklendi
                random_state=42
            ),
            
            'LightGBM': LGBMRegressor(
                n_estimators=200,          # Artırıldı
                learning_rate=0.08,        # Azaltıldı
                max_depth=8,               # Artırıldı
                num_leaves=50,             # Artırıldı
                min_child_samples=10,      # Eklendi
                subsample=0.9,             # Eklendi
                colsample_bytree=0.8,      # Eklendi
                reg_alpha=0.1,             # Regularization
                reg_lambda=0.1,            # Regularization
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            
            'CatBoost': CatBoostRegressor(
                iterations=200,            # Artırıldı
                learning_rate=0.08,        # Azaltıldı
                depth=8,                   # Artırıldı
                l2_leaf_reg=5,             # Regularization
                bootstrap_type='Bernoulli', # Eklendi
                subsample=0.9,             # Eklendi
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )
        }
        
        self.trained_models = {}
        self.model_scores = {}
    
    def _safe_mape(self, y_true, y_pred):
        """MAPE hesaplama - sıfıra bölme hatası önlenmiş"""
        # Sıfır değerleri filtrele
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        
        y_true_safe = y_true[mask]
        y_pred_safe = y_pred[mask]
        
        # MAPE hesapla
        mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100
        # Makul bir aralığa sınırla
        return min(mape, 100.0)
    
    def train_model(self, model_name, X_train, y_train, X_test=None, y_test=None, cv_strategy='kfold'):
        """
        Tek bir modeli eğit - Geliştirilmiş CV stratejisi
        
        cv_strategy: 'kfold' | 'timeseries' | 'stratified'
        """
        
        if model_name not in self.models:
            raise ValueError(f"Model bulunamadı: {model_name}")
        
        try:
            model = self.models[model_name]
            
            # Modeli eğit
            model.fit(X_train, y_train)
            
            # Cross-validation stratejisi seç
            if cv_strategy == 'timeseries':
                cv = TimeSeriesSplit(n_splits=5)
            elif cv_strategy == 'kfold':
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Cross-validation skoru
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )
            
            # Alternatif metrikler de hesapla
            neg_mse_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Test skoru (varsa)
            test_score = None
            test_rmse = None
            test_mae = None
            if X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                test_score = r2_score(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_mae = mean_absolute_error(y_test, y_pred)
            
            self.trained_models[model_name] = model
            self.model_scores[model_name] = {
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_rmse': np.sqrt(-neg_mse_scores.mean()),
                'test_score': test_score,
                'test_rmse': test_rmse,
                'test_mae': test_mae
            }
            
            return model, cv_scores.mean(), test_score
            
        except Exception as e:
            print(f"❌ {model_name} modeli eğitilirken hata: {e}")
            return None, None, None
    
    def train_all_models(self, X_train, y_train, X_test=None, y_test=None, cv_strategy='kfold'):
        """
        Tüm modelleri eğit ve karşılaştır - Geliştirilmiş
        
        cv_strategy: Cross-validation stratejisi ('kfold' veya 'timeseries')
        """
        
        results = []
        
        print(f"\n🔄 CV Strategy: {cv_strategy}")
        print(f"📊 {len(self.models)} model eğitiliyor...")
        print("-" * 60)
        
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"\n[{i}/{len(self.models)}] 🤖 {model_name} modeli eğitiliyor...")
            
            model, cv_score, test_score = self.train_model(
                model_name, X_train, y_train, X_test, y_test, cv_strategy
            )
            
            if model is not None:
                scores = self.model_scores[model_name]
                
                result = {
                    'Model': model_name,
                    'CV R2': cv_score,
                    'CV RMSE': scores['cv_rmse'],
                    'Test R2': test_score if test_score is not None else np.nan,
                    'Test RMSE': scores['test_rmse'] if scores['test_rmse'] is not None else np.nan,
                    'Test MAE': scores['test_mae'] if scores['test_mae'] is not None else np.nan
                }
                results.append(result)
                
                # Detaylı sonuç yazdırma
                if test_score is not None:
                    print(f"✅ {model_name}:")
                    print(f"   📈 CV R2: {cv_score:.4f} (±{scores['cv_std']:.4f})")
                    print(f"   🎯 Test R2: {test_score:.4f}")
                    print(f"   📉 Test RMSE: {scores['test_rmse']:.3f}")
                    print(f"   📊 Test MAE: {scores['test_mae']:.3f}")
                else:
                    print(f"✅ {model_name}:")
                    print(f"   📈 CV R2: {cv_score:.4f} (±{scores['cv_std']:.4f})")
                    print(f"   📉 CV RMSE: {scores['cv_rmse']:.3f}")
            else:
                print(f"❌ {model_name} modeli eğitilemedi!")
        
        print("\n" + "=" * 60)
        print("🏆 MODEL KARŞILAŞTIRMA TAMAMLANDI")
        print("=" * 60)
        
        results_df = pd.DataFrame(results).sort_values('CV R2', ascending=False)
        
        if len(results_df) > 0:
            print(f"\n🥇 En iyi model: {results_df.iloc[0]['Model']} (CV R2: {results_df.iloc[0]['CV R2']:.4f})")
        
        return results_df
    
    def get_best_model(self):
        """En iyi modeli döndür"""
        
        if not self.model_scores:
            return None, None
        
        # CV skoruna göre en iyi model
        best_model_name = max(
            self.model_scores.keys(),
            key=lambda x: self.model_scores[x]['cv_score']
        )
        
        return best_model_name, self.trained_models[best_model_name]
    
    def predict(self, model_name, X):
        """Belirli bir model ile tahmin yap"""
        
        if model_name not in self.trained_models:
            raise ValueError(f"Eğitilmiş model bulunamadı: {model_name}")
        
        return self.trained_models[model_name].predict(X)
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Model performansını değerlendir"""
        
        if model_name not in self.trained_models:
            raise ValueError(f"Eğitilmiş model bulunamadı: {model_name}")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'R2 Score': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': self._safe_mape(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def get_feature_importance(self, model_name, feature_names):
        """Özellik önem skorlarını al"""
        
        if model_name not in self.trained_models:
            return None
        
        model = self.trained_models[model_name]
        
        # Feature importance destekleyen modeller
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None