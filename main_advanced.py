"""
Gelişmiş Finansal Risk Tahmin Sistemi - Ana Program
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from src.loader import load_data
from src.automl_system import AutoMLPipeline
from src.advanced_models import AdvancedRiskModels
from src.feature_engineering import AdvancedFeatureEngineering
from src.model_evaluation import ModelEvaluator
from src.hyperparameter_tuning import HyperparameterOptimizer


def create_directories():
    """Gerekli dizinleri oluştur"""
    directories = ['models', 'models/automl', 'reports', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Dizinler oluşturuldu")


def run_quick_training():
    """Hızlı model eğitimi (hyperparameter tuning olmadan)"""
    
    print("\n" + "="*60)
    print("⚡ HIZLI MODEL EĞİTİMİ")
    print("="*60)
    
    # Veri yükle
    df = load_data("data/birlesik_risk_verisi.csv")
    
    # AutoML pipeline
    automl = AutoMLPipeline(optimize_hyperparams=False)
    results = automl.run_automl(df)
    
    # Yeni veri tahmini
    print("\n📊 Yeni müşteri verileri için tahmin yapılıyor...")
    predictions = automl.predict_new_data("data/yeni_musteri.csv")
    print("\nTahmin Sonuçları:")
    print(predictions.head(10))
    
    # Tahminleri kaydet
    predictions.to_csv("reports/predictions_quick.csv", index=False)
    print("✅ Tahminler kaydedildi: reports/predictions_quick.csv")
    
    return results


def run_optimized_training():
    """Optimize edilmiş model eğitimi (hyperparameter tuning ile)"""
    
    print("\n" + "="*60)
    print("🔧 OPTİMİZE EDİLMİŞ MODEL EĞİTİMİ")
    print("="*60)
    
    # Veri yükle
    df = load_data("data/birlesik_risk_verisi.csv")
    
    # AutoML pipeline (optimize edilmiş)
    automl = AutoMLPipeline(optimize_hyperparams=True, n_trials=30)
    results = automl.run_automl(df)
    
    # Yeni veri tahmini
    print("\n📊 Yeni müşteri verileri için tahmin yapılıyor...")
    predictions = automl.predict_new_data("data/yeni_musteri.csv")
    print("\nTahmin Sonuçları:")
    print(predictions.head(10))
    
    # Tahminleri kaydet
    predictions.to_csv("reports/predictions_optimized.csv", index=False)
    print("✅ Tahminler kaydedildi: reports/predictions_optimized.csv")
    
    return results


def run_custom_pipeline():
    """Özelleştirilmiş pipeline"""
    
    print("\n" + "="*60)
    print("🎯 ÖZELLEŞTİRİLMİŞ PİPELINE")
    print("="*60)
    
    # 1. Veri yükle
    df = load_data("data/birlesik_risk_verisi.csv")
    
    # 2. Feature engineering
    print("\n📊 Feature engineering yapılıyor...")
    feature_engineer = AdvancedFeatureEngineering()
    df = feature_engineer.create_advanced_features(df)
    
    # 3. Target oluştur
    from src.risk_calculator import calculate_risk_from_dataframe
    df['RiskScore'] = calculate_risk_from_dataframe(df)
    
    # 4. Özellik seçimi
    from src.config import config
    exclude_cols = ['RiskScore'] + config.SYSTEM_COLUMNS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0)
    y = df['RiskScore']
    
    # 5. Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Scaling
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # 7. Polynomial features
    print("📐 Polynomial features oluşturuluyor...")
    X_train_poly = feature_engineer.create_polynomial_features(X_train_scaled, degree=2)
    X_test_poly = feature_engineer.create_polynomial_features(X_test_scaled, degree=2)
    
    # 8. Feature selection
    print("🎯 En iyi özellikler seçiliyor...")
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    
    # Özellik sayısını sınırla (max 50)
    k_features = min(50, X_train_poly.shape[1])
    selector = SelectKBest(score_func=mutual_info_regression, k=k_features)
    X_train_selected = selector.fit_transform(X_train_poly, y_train)
    X_test_selected = selector.transform(X_test_poly)
    
    selected_features = X_train_poly.columns[selector.get_support()].tolist()
    X_train_final = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    X_test_final = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
    
    print(f"✅ {k_features} özellik seçildi")
    
    # 9. Model eğitimi
    print("\n📊 Gelişmiş modeller eğitiliyor...")
    model_trainer = AdvancedRiskModels()
    results_df = model_trainer.train_all_models(X_train_final, y_train, X_test_final, y_test)
    
    # 10. En iyi model
    best_model_name, best_model = model_trainer.get_best_model()
    print(f"\n🏆 En iyi model: {best_model_name}")
    
    # 11. Model değerlendirme
    evaluator = ModelEvaluator()
    
    # Görselleştirmeler
    print("\n📈 Görselleştirmeler oluşturuluyor...")
    
    # Model karşılaştırması
    evaluator.plot_model_comparison(results_df)
    
    # En iyi model için tahmin analizi
    y_pred_test = best_model.predict(X_test_final)
    evaluator.plot_predictions(y_test, y_pred_test, title=f"{best_model_name} Tahmin Analizi")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance_dict = {best_model_name: pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)}
        evaluator.plot_feature_importance(importance_dict)
    
    # Learning curves
    evaluator.plot_learning_curves(best_model, X_train_final, y_train)
    
    # Error distribution
    evaluator.plot_error_distribution(y_test, y_pred_test, best_model_name)
    
    # 12. Rapor oluştur
    evaluation_report = evaluator.create_evaluation_report(results_df)
    
    print("\n✅ Özelleştirilmiş pipeline tamamlandı!")
    
    return {
        'results_df': results_df,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features
    }


def main():
    """Ana program"""
    
    print("\n" + "="*70)
    print("🚀 GELİŞMİŞ FİNANSAL RİSK TAHMİN SİSTEMİ")
    print("="*70)
    
    # Dizinleri oluştur
    create_directories()
    
    print("\nLütfen bir seçenek seçin:")
    print("1. Hızlı Model Eğitimi (Hyperparameter tuning olmadan)")
    print("2. Optimize Edilmiş Model Eğitimi (Hyperparameter tuning ile)")
    print("3. Özelleştirilmiş Pipeline (Detaylı analiz ve görselleştirme)")
    print("4. Tüm Seçenekleri Çalıştır")
    
    try:
        choice = input("\nSeçiminiz (1-4): ").strip()
        
        if choice == "1":
            run_quick_training()
        elif choice == "2":
            run_optimized_training()
        elif choice == "3":
            run_custom_pipeline()
        elif choice == "4":
            print("\n🔄 Tüm seçenekler sırayla çalıştırılıyor...")
            run_quick_training()
            run_optimized_training()
            run_custom_pipeline()
        else:
            print("❌ Geçersiz seçim. Program sonlandırılıyor.")
            return
        
        print("\n" + "="*70)
        print("✅ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
        print("="*70)
        print("\n📁 Sonuçlar:")
        print("   - Modeller: models/ dizininde")
        print("   - Raporlar: reports/ dizininde")
        print("   - Görselleştirmeler: plots/ dizininde")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Program kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()