"""
Model değerlendirme ve görselleştirme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """Model değerlendirme ve görselleştirme"""
    
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
            'train_mape': mean_absolute_percentage_error(y_train, y_pred_train),
            'test_mape': mean_absolute_percentage_error(y_test, y_pred_test),
            'overfitting_score': abs(r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test))
        }
        
        self.evaluation_results[model_name] = metrics
        
        return metrics, y_pred_train, y_pred_test
    
    def plot_predictions(self, y_true, y_pred, title="Gerçek vs Tahmin"):
        """Tahmin görselleştirmesi"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Gerçek Değerler')
        axes[0].set_ylabel('Tahmin Değerleri')
        axes[0].set_title(f'{title} - Scatter Plot')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Tahmin Değerleri')
        axes[1].set_ylabel('Residuals (Gerçek - Tahmin)')
        axes[1].set_title(f'{title} - Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/prediction_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_model_comparison(self, results_df):
        """Model karşılaştırma görselleştirmesi"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # R2 Score karşılaştırması
        ax = axes[0, 0]
        x = np.arange(len(results_df))
        width = 0.35
        ax.bar(x - width/2, results_df['Train_R2'], width, label='Train R2', alpha=0.8)
        ax.bar(x + width/2, results_df['Test_R2'], width, label='Test R2', alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('R2 Score')
        ax.set_title('R2 Score Karşılaştırması')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MSE karşılaştırması
        ax = axes[0, 1]
        ax.bar(x - width/2, results_df['Train_MSE'], width, label='Train MSE', alpha=0.8)
        ax.bar(x + width/2, results_df['Test_MSE'], width, label='Test MSE', alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('MSE')
        ax.set_title('MSE Karşılaştırması')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MAE karşılaştırması
        ax = axes[0, 2]
        ax.bar(x - width/2, results_df['Train_MAE'], width, label='Train MAE', alpha=0.8)
        ax.bar(x + width/2, results_df['Test_MAE'], width, label='Test MAE', alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE')
        ax.set_title('MAE Karşılaştırması')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cross-validation R2
        ax = axes[1, 0]
        ax.errorbar(x, results_df['CV_R2_Mean'], yerr=results_df['CV_R2_Std'], 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax.set_xlabel('Model')
        ax.set_ylabel('CV R2 Score')
        ax.set_title('Cross-Validation R2 Score')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Overfitting Score
        ax = axes[1, 1]
        colors = ['green' if score < 0.1 else 'orange' if score < 0.2 else 'red' 
                  for score in results_df['Overfitting_Score']]
        ax.bar(x, results_df['Overfitting_Score'], color=colors, alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('Overfitting Score')
        ax.set_title('Overfitting Score (Lower is Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good')
        ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Model sıralaması
        ax = axes[1, 2]
        top_models = results_df.nlargest(5, 'Test_R2')
        ax.barh(range(len(top_models)), top_models['Test_R2'], color='skyblue', alpha=0.8)
        ax.set_yticks(range(len(top_models)))
        ax.set_yticklabels(top_models['Model'])
        ax.set_xlabel('Test R2 Score')
        ax.set_title('En İyi 5 Model')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, importance_dict, top_n=15):
        """Feature importance görselleştirmesi"""
        
        n_models = len(importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance_df) in enumerate(importance_dict.items()):
            ax = axes[idx]
            
            # Top N özelliği al
            top_features = importance_df.head(top_n)
            
            # Barplot
            ax.barh(range(len(top_features)), top_features['importance'], 
                    color='steelblue', alpha=0.8)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance Score')
            ax.set_title(f'{model_name} - Top {top_n} Features')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Değerleri bar üzerine yaz
            for i, v in enumerate(top_features['importance']):
                ax.text(v, i, f'{v:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_learning_curves(self, model, X, y, cv=5):
        """Öğrenme eğrisi görselleştirmesi"""
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ortalama ve standart sapma
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation Score')
        
        # Confidence intervals
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='green')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('R2 Score')
        ax.set_title('Learning Curves')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/learning_curves.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig
    
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
    
    def plot_error_distribution(self, y_true, y_pred, model_name="Model"):
        """Hata dağılımı görselleştirmesi"""
        
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name} - Error Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1])
        axes[1].set_title(f'{model_name} - Q-Q Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Box plot
        axes[2].boxplot(errors, vert=True)
        axes[2].set_ylabel('Prediction Error')
        axes[2].set_title(f'{model_name} - Error Box Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_error_distribution.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig