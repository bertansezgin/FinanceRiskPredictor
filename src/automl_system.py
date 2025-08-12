"""
AutoML System - Otomatik makine öğrenmesi pipeline'ı
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

from src.feature_engineering import AdvancedFeatureEngineering
from src.advanced_models import AdvancedRiskModels
from src.hyperparameter_tuning import HyperparameterOptimizer
from src.model_evaluation import ModelEvaluator


class AutoMLPipeline:
    """Tam otomatik ML pipeline"""
    
    def __init__(self, task_type='regression', optimize_hyperparams=True, n_trials=30):
        self.task_type = task_type
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        
        self.feature_engineer = AdvancedFeatureEngineering()
        self.model_trainer = AdvancedRiskModels()
        self.hyperopt = HyperparameterOptimizer(n_trials=n_trials)
        self.evaluator = ModelEvaluator()
        
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.results = {}
        
    def prepare_data(self, df, target_col=None):
        """Veri hazırlama"""
        
        print("📊 Veri hazırlanıyor...")
        
        # Feature engineering
        df = self.feature_engineer.create_advanced_features(df)
        
        # Target değişkeni oluştur (eğer verilmediyse)
        if target_col is None:
            from src.risk_calculator import calculate_risk_from_dataframe
            # Risk skoru hesapla
            df['RiskScore'] = calculate_risk_from_dataframe(df)
            target_col = 'RiskScore'
        
        # Özellik seçimi
        from src.config import config
        exclude_cols = [target_col] + config.SYSTEM_COLUMNS
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Sadece sayısal özellikleri al
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        self.feature_names = numeric_cols
        
        print(f"✅ Veri hazırlandı: {X.shape[0]} satır, {X.shape[1]} özellik")
        
        return X, y
    
    def run_automl(self, df, target_col=None, test_size=0.2):
        """AutoML pipeline'ı çalıştır"""
        
        print("="*60)
        print("🚀 AutoML Pipeline Başlıyor...")
        print("="*60)
        
        # 1. Veri hazırlama
        X, y = self.prepare_data(df, target_col)
        
        # 2. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"✅ Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # 3. Feature scaling
        print("🔄 Özellikler ölçeklendiriliyor...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # DataFrame'e çevir
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # 4. Model eğitimi
        if self.optimize_hyperparams:
            print("\n🔧 Hyperparameter optimizasyonu yapılıyor...")
            optimization_results = self.hyperopt.optimize_all_models(X_train_scaled, y_train)
            
            # En iyi modelleri kullan
            self.best_model_name = max(optimization_results.keys(), 
                                       key=lambda x: optimization_results[x]['cv_score'])
            self.best_model = self.hyperopt.best_models[self.best_model_name]
            
            print(f"\n✅ Optimizasyon tamamlandı. En iyi model: {self.best_model_name}")
            
        else:
            print("\n📊 Modeller eğitiliyor...")
            results_df = self.model_trainer.train_all_models(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            # En iyi modeli seç
            self.best_model_name, self.best_model = self.model_trainer.get_best_model()
            print(f"\n✅ En iyi model: {self.best_model_name}")
        
        # 5. Model değerlendirme
        print("\n📈 Model değerlendiriliyor...")
        metrics, y_pred_train, y_pred_test = self.evaluator.evaluate_model(
            self.best_model, X_train_scaled, y_train, X_test_scaled, y_test, 
            self.best_model_name
        )
        
        # 6. Sonuçları kaydet
        self.results = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'metrics': metrics,
            'scaler': scaler,
            'feature_names': self.feature_names,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }
        
        # 7. Model ve scaler'ı kaydet
        self.save_model(self.best_model, scaler)
        
        # 8. Rapor oluştur
        self.generate_report()
        
        print("\n" + "="*60)
        print("✅ AutoML Pipeline Tamamlandı!")
        print("="*60)
        
        return self.results
    
    def save_model(self, model, scaler):
        """Modeli ve scaler'ı kaydet"""
        
        # Dizin oluştur
        os.makedirs('models/automl', exist_ok=True)
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Model kaydet
        model_path = f'models/automl/best_model_{self.best_model_name}_{timestamp}.pkl'
        joblib.dump(model, model_path)
        print(f"✅ Model kaydedildi: {model_path}")
        
        # Scaler kaydet
        scaler_path = f'models/automl/scaler_{timestamp}.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler kaydedildi: {scaler_path}")
        
        # Feature names kaydet
        features_path = f'models/automl/features_{timestamp}.pkl'
        joblib.dump(self.feature_names, features_path)
        print(f"✅ Feature names kaydedildi: {features_path}")
        
        # Model bilgilerini kaydet
        model_info = {
            'model_name': self.best_model_name,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path,
            'metrics': self.results.get('metrics', {}),
            'timestamp': timestamp,
            'n_features': len(self.feature_names)
        }
        
        import json
        info_path = f'models/automl/model_info_{timestamp}.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4, default=str)
        print(f"✅ Model bilgileri kaydedildi: {info_path}")
        
    def generate_report(self):
        """Detaylı rapor oluştur"""
        
        # Dizin oluştur
        os.makedirs('reports', exist_ok=True)
        
        report = []
        report.append("="*70)
        report.append("AUTOML PİPELINE RAPORU")
        report.append("="*70)
        report.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model bilgileri
        report.append("📊 MODEL BİLGİLERİ")
        report.append("-"*50)
        report.append(f"En İyi Model: {self.best_model_name}")
        report.append(f"Özellik Sayısı: {len(self.feature_names)}")
        report.append(f"Eğitim Seti Boyutu: {self.results['X_train'].shape}")
        report.append(f"Test Seti Boyutu: {self.results['X_test'].shape}")
        report.append("")
        
        # Performans metrikleri
        metrics = self.results['metrics']
        report.append("📈 PERFORMANS METRİKLERİ")
        report.append("-"*50)
        report.append(f"Test R2 Score: {metrics['test_r2']:.4f}")
        report.append(f"Test RMSE: {metrics['test_rmse']:.4f}")
        report.append(f"Test MAE: {metrics['test_mae']:.4f}")
        report.append(f"Test MAPE: {metrics['test_mape']:.4f}")
        report.append("")
        report.append(f"Train R2 Score: {metrics['train_r2']:.4f}")
        report.append(f"Train RMSE: {metrics['train_rmse']:.4f}")
        report.append(f"Overfitting Score: {metrics['overfitting_score']:.4f}")
        report.append("")
        
        # Özellik önemi (eğer varsa)
        if hasattr(self.best_model, 'feature_importances_'):
            report.append("🎯 EN ÖNEMLİ 10 ÖZELLİK")
            report.append("-"*50)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for idx, row in importance_df.head(10).iterrows():
                report.append(f"{row['feature']:<30} {row['importance']:.4f}")
            report.append("")
        
        # Hyperparameter bilgileri
        if self.optimize_hyperparams and self.hyperopt.best_params:
            report.append("🔧 OPTİMİZE EDİLMİŞ HYPERPARAMETRELER")
            report.append("-"*50)
            
            if self.best_model_name in self.hyperopt.best_params:
                for param, value in self.hyperopt.best_params[self.best_model_name].items():
                    report.append(f"{param:<25} {value}")
            report.append("")
        
        report.append("="*70)
        
        # Raporu kaydet
        report_text = "\n".join(report)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'reports/automl_report_{timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n📄 Rapor kaydedildi: {report_path}")
        
        # Raporu ekrana da yazdır
        print("\n" + report_text)
        
        return report_text
    
    def predict_new_data(self, new_data_path):
        """Yeni veri için tahmin yap"""
        
        if self.best_model is None:
            raise ValueError("Önce model eğitilmeli!")
        
        # Veriyi yükle
        new_df = pd.read_csv(new_data_path)
        
        # Feature engineering
        new_df = self.feature_engineer.create_advanced_features(new_df)
        
        # Aynı özellikleri seç
        X_new = new_df[self.feature_names].fillna(0)
        
        # Scale
        if 'scaler' in self.results:
            X_new_scaled = self.results['scaler'].transform(X_new)
        else:
            X_new_scaled = X_new
        
        # Tahmin
        predictions = self.best_model.predict(X_new_scaled)
        
        # Sonuçları DataFrame'e ekle
        new_df['PredictedRiskScore'] = predictions
        new_df['PredictedRiskScore'] = new_df['PredictedRiskScore'].clip(0, 100)
        
        # Risk kategorisi
        new_df['RiskCategory'] = pd.cut(
            new_df['PredictedRiskScore'],
            bins=[0, 25, 50, 75, 100],
            labels=['Yüksek Risk', 'Orta Risk', 'Düşük Risk', 'Çok Düşük Risk']
        )
        
        return new_df[['ProjectId', 'PredictedRiskScore', 'RiskCategory']]