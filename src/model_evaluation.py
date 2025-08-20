"""
Model değerlendirme ve rapor oluşturma
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from src.utils import safe_mape


class ModelEvaluator:
    """Model değerlendirme ve rapor oluşturma"""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name="Model"):
        """Modeli değerlendir ve metrikleri hesapla"""

        # Tahminler
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrikler
        metrics = {
            'model_name': model_name,
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mape': safe_mape(y_train, y_pred_train),
            'test_mape': safe_mape(y_test, y_pred_test),
            'overfitting_score': abs(r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test))
        }

        self.evaluation_results[model_name] = metrics

        return metrics, y_pred_train, y_pred_test

    def create_evaluation_report(self, results_df):
        """Değerlendirme raporu oluştur"""

        report = []
        report.append("="*60)
        report.append("MODEL DEĞERLENDİRME RAPORU")
        report.append("="*60)
        report.append("")

        # En iyi model
        best_model = results_df.iloc[0]
        report.append(f"🏆 EN İYİ MODEL: {best_model['Model']}")
        report.append(f"   Test R2 Score: {best_model['Test_R2']:.4f}")
        report.append(f"   Test RMSE: {np.sqrt(best_model['Test_MSE']):.4f}")
        report.append(f"   CV R2 Mean: {best_model['CV_R2_Mean']:.4f} (±{best_model['CV_R2_Std']:.4f})")
        report.append("")

        # Model sıralaması
        report.append("📊 MODEL SIRALAMASI (Test R2'ye göre):")
        report.append("-"*40)
        for idx, row in results_df.iterrows():
            report.append(f"{idx+1}. {row['Model']:<20} R2: {row['Test_R2']:.4f}")
        report.append("")

        # Overfitting analizi
        report.append("⚠️ OVERFITTING ANALİZİ:")
        report.append("-"*40)
        for _, row in results_df.iterrows():
            overfitting = row['Overfitting_Score']
            if overfitting < 0.05:
                status = "✅ Mükemmel"
            elif overfitting < 0.1:
                status = "✅ İyi"
            elif overfitting < 0.2:
                status = "⚠️ Kabul edilebilir"
            else:
                status = "❌ Overfitting var"

            report.append(f"{row['Model']:<20} {status} (Score: {overfitting:.4f})")

        report.append("")
        report.append("="*60)

        # Raporu yazdır ve kaydet
        report_text = "\n".join(report)
        print(report_text)

        with open('reports/model_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        return report_text