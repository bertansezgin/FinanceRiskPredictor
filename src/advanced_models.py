"""
Gelişmiş makine öğrenmesi modelleri
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class AdvancedRiskModels:
    """Gelişmiş risk tahmin modelleri"""
    
    def __init__(self):
        self.models = {
            # Linear Models
            'LinearRegression': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=1000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
            
            # Tree-based Models
            'DecisionTree': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            ),
            
            # Advanced Boosting Models
            'XGBoost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            'CatBoost': CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                random_state=42,
                verbose=False
            ),
            
            # Other Models
            'KNN': KNeighborsRegressor(
                n_neighbors=10,
                weights='distance',
                algorithm='auto'
            ),
            'SVM': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            ),
            'NeuralNetwork': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        self.trained_models = {}
        self.model_scores = {}
        
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Tüm modelleri eğit ve performanslarını değerlendir"""
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n📊 {name} modeli eğitiliyor...")
            
            try:
                # Model eğitimi
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Tahminler
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrikler
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Cross-validation (5-fold)
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='r2',
                    n_jobs=-1
                )
                
                result = {
                    'Model': name,
                    'Train_MSE': train_mse,
                    'Test_MSE': test_mse,
                    'Train_R2': train_r2,
                    'Test_R2': test_r2,
                    'Train_MAE': train_mae,
                    'Test_MAE': test_mae,
                    'CV_R2_Mean': cv_scores.mean(),
                    'CV_R2_Std': cv_scores.std(),
                    'Overfitting_Score': abs(train_r2 - test_r2)
                }
                
                results.append(result)
                self.model_scores[name] = result
                
                print(f"✅ {name} - Test R2: {test_r2:.4f}, CV R2: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"❌ {name} modeli eğitilirken hata: {str(e)}")
                continue
        
        # Sonuçları DataFrame'e çevir
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test_R2', ascending=False)
        
        return results_df
    
    def get_best_model(self, metric='Test_R2'):
        """En iyi modeli döndür"""
        if not self.model_scores:
            raise ValueError("Henüz hiçbir model eğitilmedi!")
        
        best_model_name = max(self.model_scores.keys(), 
                              key=lambda x: self.model_scores[x][metric])
        
        return best_model_name, self.trained_models[best_model_name]
    
    def get_ensemble_predictions(self, X, models_to_use=None, weights=None):
        """Ensemble tahmin yap"""
        if models_to_use is None:
            # En iyi 3 modeli kullan
            sorted_models = sorted(self.model_scores.keys(), 
                                   key=lambda x: self.model_scores[x]['Test_R2'], 
                                   reverse=True)[:3]
            models_to_use = sorted_models
        
        if weights is None:
            weights = [1/len(models_to_use)] * len(models_to_use)
        
        predictions = []
        for model_name in models_to_use:
            if model_name in self.trained_models:
                pred = self.trained_models[model_name].predict(X)
                predictions.append(pred)
        
        # Ağırlıklı ortalama
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def get_feature_importance(self, feature_names):
        """Feature importance analizi"""
        importance_dict = {}
        
        # Tree-based modeller için feature importance
        tree_models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
        
        for model_name in tree_models:
            if model_name in self.trained_models:
                model = self.trained_models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importance_dict[model_name] = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
        
        return importance_dict