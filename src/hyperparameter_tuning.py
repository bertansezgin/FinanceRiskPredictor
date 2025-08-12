"""
Hyperparameter Optimization with Optuna
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Optuna ile hyperparameter optimizasyonu"""
    
    def __init__(self, n_trials=50, cv_folds=5, random_state=42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = {}
        self.best_models = {}
        self.study_results = {}
        
    def optimize_random_forest(self, X_train, y_train):
        """Random Forest için hyperparameter optimizasyonu"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**params)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='r2',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='RandomForest')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['RandomForest'] = study.best_params
        self.study_results['RandomForest'] = study
        
        # En iyi parametrelerle modeli eğit
        best_model = RandomForestRegressor(**study.best_params, random_state=self.random_state, n_jobs=-1)
        best_model.fit(X_train, y_train)
        self.best_models['RandomForest'] = best_model
        
        print(f"✅ RandomForest optimizasyonu tamamlandı. Best CV R2: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    def optimize_xgboost(self, X_train, y_train):
        """XGBoost için hyperparameter optimizasyonu"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = XGBRegressor(**params)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='r2',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='XGBoost')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['XGBoost'] = study.best_params
        self.study_results['XGBoost'] = study
        
        # En iyi parametrelerle modeli eğit
        best_model = XGBRegressor(**study.best_params, random_state=self.random_state, n_jobs=-1)
        best_model.fit(X_train, y_train)
        self.best_models['XGBoost'] = best_model
        
        print(f"✅ XGBoost optimizasyonu tamamlandı. Best CV R2: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    def optimize_lightgbm(self, X_train, y_train):
        """LightGBM için hyperparameter optimizasyonu"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
            
            model = LGBMRegressor(**params)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='r2',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='LightGBM')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['LightGBM'] = study.best_params
        self.study_results['LightGBM'] = study
        
        # En iyi parametrelerle modeli eğit
        best_model = LGBMRegressor(**study.best_params, random_state=self.random_state, n_jobs=-1, verbosity=-1)
        best_model.fit(X_train, y_train)
        self.best_models['LightGBM'] = best_model
        
        print(f"✅ LightGBM optimizasyonu tamamlandı. Best CV R2: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    def optimize_catboost(self, X_train, y_train):
        """CatBoost için hyperparameter optimizasyonu"""
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 1),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': self.random_state,
                'verbose': False
            }
            
            model = CatBoostRegressor(**params)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='r2',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='CatBoost')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['CatBoost'] = study.best_params
        self.study_results['CatBoost'] = study
        
        # En iyi parametrelerle modeli eğit
        best_model = CatBoostRegressor(**study.best_params, random_state=self.random_state, verbose=False)
        best_model.fit(X_train, y_train)
        self.best_models['CatBoost'] = best_model
        
        print(f"✅ CatBoost optimizasyonu tamamlandı. Best CV R2: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    def optimize_gradient_boosting(self, X_train, y_train):
        """Gradient Boosting için hyperparameter optimizasyonu"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
            
            model = GradientBoostingRegressor(**params)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='r2',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='GradientBoosting')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['GradientBoosting'] = study.best_params
        self.study_results['GradientBoosting'] = study
        
        # En iyi parametrelerle modeli eğit
        best_model = GradientBoostingRegressor(**study.best_params, random_state=self.random_state)
        best_model.fit(X_train, y_train)
        self.best_models['GradientBoosting'] = best_model
        
        print(f"✅ GradientBoosting optimizasyonu tamamlandı. Best CV R2: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    def optimize_all_models(self, X_train, y_train):
        """Tüm modeller için optimizasyon yap"""
        
        print("🔧 Hyperparameter optimizasyonu başlıyor...")
        print(f"   Trials: {self.n_trials}, CV Folds: {self.cv_folds}")
        print("-" * 50)
        
        results = {}
        
        # RandomForest
        try:
            params, score = self.optimize_random_forest(X_train, y_train)
            results['RandomForest'] = {'params': params, 'cv_score': score}
        except Exception as e:
            print(f"❌ RandomForest optimizasyonu başarısız: {str(e)}")
        
        # XGBoost
        try:
            params, score = self.optimize_xgboost(X_train, y_train)
            results['XGBoost'] = {'params': params, 'cv_score': score}
        except Exception as e:
            print(f"❌ XGBoost optimizasyonu başarısız: {str(e)}")
        
        # LightGBM
        try:
            params, score = self.optimize_lightgbm(X_train, y_train)
            results['LightGBM'] = {'params': params, 'cv_score': score}
        except Exception as e:
            print(f"❌ LightGBM optimizasyonu başarısız: {str(e)}")
        
        # CatBoost
        try:
            params, score = self.optimize_catboost(X_train, y_train)
            results['CatBoost'] = {'params': params, 'cv_score': score}
        except Exception as e:
            print(f"❌ CatBoost optimizasyonu başarısız: {str(e)}")
        
        # GradientBoosting
        try:
            params, score = self.optimize_gradient_boosting(X_train, y_train)
            results['GradientBoosting'] = {'params': params, 'cv_score': score}
        except Exception as e:
            print(f"❌ GradientBoosting optimizasyonu başarısız: {str(e)}")
        
        print("-" * 50)
        print("✅ Optimizasyon tamamlandı!")
        
        # En iyi modeli bul
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_score'])
        print(f"\n🏆 En iyi model: {best_model_name}")
        print(f"   CV R2 Score: {results[best_model_name]['cv_score']:.4f}")
        
        return results
    
    def get_optimization_report(self):
        """Optimizasyon raporu oluştur"""
        
        report = []
        report.append("="*60)
        report.append("HYPERPARAMETER OPTİMİZASYON RAPORU")
        report.append("="*60)
        report.append("")
        
        for model_name, params in self.best_params.items():
            report.append(f"📊 {model_name}")
            report.append("-"*40)
            
            if model_name in self.study_results:
                study = self.study_results[model_name]
                report.append(f"Best CV Score: {study.best_value:.4f}")
                report.append(f"Number of trials: {len(study.trials)}")
                report.append("Best Parameters:")
                
                for param_name, param_value in params.items():
                    report.append(f"  - {param_name}: {param_value}")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        # Raporu kaydet
        with open('reports/hyperparameter_optimization_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def visualize_optimization_history(self):
        """Optimizasyon geçmişini görselleştir"""
        
        import matplotlib.pyplot as plt
        
        n_models = len(self.study_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, study) in enumerate(self.study_results.items()):
            ax = axes[idx]
            
            # Trial değerleri
            trial_values = [trial.value for trial in study.trials]
            
            # Plot
            ax.plot(trial_values, marker='o', linestyle='-', alpha=0.7)
            ax.axhline(y=study.best_value, color='r', linestyle='--', 
                      label=f'Best: {study.best_value:.4f}')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('CV R2 Score')
            ax.set_title(f'{model_name} Optimization History')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/optimization_history.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def visualize_param_importance(self):
        """Parametre önem analizi"""
        
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_param_importances
        
        for model_name, study in self.study_results.items():
            try:
                fig = plot_param_importances(study)
                fig.update_layout(title=f'{model_name} - Parameter Importance')
                fig.write_html(f'plots/{model_name}_param_importance.html')
                print(f"✅ {model_name} parametre önem grafiği kaydedildi")
            except Exception as e:
                print(f"❌ {model_name} parametre önem grafiği oluşturulamadı: {str(e)}")