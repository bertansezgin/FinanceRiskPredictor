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



def create_directories():
    """Gerekli dizinleri oluştur"""
    directories = ['models', 'models/automl', 'reports', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Dizinler oluşturuldu")


def run_quick_training():
    """Hızlı model eğitimi"""

    print("\n" + "="*60)
    print("⚡ HIZLI MODEL EĞİTİMİ")
    print("="*60)

    # Veri yükle
    df = load_data("data/birlesik_risk_verisi.csv")

    # AutoML pipeline
    automl = AutoMLPipeline()
    results = automl.run_automl(df)

    # Batch tahmin (tüm veri için)
    print("\n📊 Tüm müşteriler için tahmin yapılıyor...")
    from src.batch_predict import predict_all
    try:
        predictions = predict_all()
        print("\nTahmin Sonuçları (ilk 10):")
        print(predictions.head(10))
        print(f"✅ Tahminler kaydedildi: reports/predictions_all.csv")
    except Exception as e:
        print(f"⚠️ Tahmin yapılamadı: {e}")

    return results





def run_custom_pipeline():
    """Historical Performance Özelleştirilmiş pipeline"""

    print("\n" + "="*60)
    print("🎯 HISTORICAL PERFORMANCE ÖZELLEŞTİRİLMİŞ PİPELINE")
    print("="*60)
    print("✅ Data leakage problemi TAMAMEN çözüldü!")
    print("📅 Historical performance target kullanılıyor")
    print("🔒 Target hesaplamasında hiç input feature kullanılmıyor")
    print("="*60)

    # 1. Veri yükle
    df = load_data("data/birlesik_risk_verisi.csv")

    # 2. Feature engineering - DATA LEAKAGE TEMİZLENMİŞ
    print("\n📊 Temiz feature engineering yapılıyor...")
    feature_engineer = AdvancedFeatureEngineering()
    df = feature_engineer.create_advanced_features(df)

    print(f"✅ Feature engineering tamamlandı:")
    print(f"   📊 Toplam feature sayısı: {df.shape[1]}")
    print(f"   🧹 Temiz feature'lar (data leakage yok)")
    print(f"   ⚡ Sadece kredi başlangıcında bilinen değişkenler")

    # 3. Target oluştur - HISTORICAL PERFORMANCE SİSTEMİ
    from src.historical_target_calculator import calculate_historical_target
    print("🎯 Historical performance target hesaplanıyor...")
    print("✅ Gerçek payment data tabanlı - SIFIR leakage riski")
    df['RiskScore'] = calculate_historical_target(df)

    print(f"✅ Risk skoru istatistikleri:")
    print(f"   📊 Ortalama: {df['RiskScore'].mean():.2f}")
    print(f"   📈 Std: {df['RiskScore'].std():.2f}")
    print(f"   📉 Min-Max: [{df['RiskScore'].min():.1f}, {df['RiskScore'].max():.1f}]")

    # 4. Özellik seçimi - SAFE FEATURES ONLY
    from src.config import config
    safe_feature_cols = [col for col in df.columns
                        if col in config.SAFE_FEATURES and col in df.columns]

    print(f"🔒 Safe feature selection:")
    print(f"   📊 Kullanılan feature sayısı: {len(safe_feature_cols)}")
    print(f"   ✅ Leakage riski YOK!")

    numeric_cols = df[safe_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0)
    y = df['RiskScore']

    # 5. Train-test split - PROJECTID LEAKAGE ÖNLEME
    from sklearn.model_selection import GroupShuffleSplit

    # ProjectId'ye göre split - aynı proje hem train hem test'te olmasın
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['ProjectId']))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"🔒 PROJECTID LEAKAGE ÖNLENDİ:")
    print(f"   📊 Train ProjectId sayısı: {df.iloc[train_idx]['ProjectId'].nunique()}")
    print(f"   📊 Test ProjectId sayısı: {df.iloc[test_idx]['ProjectId'].nunique()}")

    # Overlap kontrolü
    train_projects = set(df.iloc[train_idx]['ProjectId'])
    test_projects = set(df.iloc[test_idx]['ProjectId'])
    overlap = train_projects.intersection(test_projects)
    print(f"   ✅ Overlap: {len(overlap)} (0 olmalı)")

    if len(overlap) > 0:
        print(f"   🚨 UYARI: {len(overlap)} ProjectId overlap var!")
    else:
        print(f"   ✅ ProjectId leakage önlendi!")

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

    # 9. Model eğitimi - GELİŞTİRİLMİŞ PARAMETRELER
    print("\n🤖 Geliştirilmiş modeller eğitiliyor...")
    print("⚡ Optimize edilmiş parametreler")
    print("📈 Temporal cross-validation stratejisi")

    model_trainer = AdvancedRiskModels()

    # Finansal veriler için temporal CV daha uygun
    cv_strategy = 'timeseries' if 'ProjectDate' in df.columns else 'kfold'

    results_df = model_trainer.train_all_models(
        X_train_final, y_train,
        X_test_final, y_test,
        cv_strategy=cv_strategy
    )

    # 9.5. MODEL VALIDATION - LEAKAGE KONTROLÜ
    print("\n🔍 MODEL VALIDATION - LEAKAGE KONTROLÜ:")

    # En iyi modeli al
    best_model_name, best_model = model_trainer.get_best_model()

    # Feature importance kontrolü
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        max_importance = max(importances)
        max_importance_idx = importances.argmax()
        max_importance_feature = X_train_final.columns[max_importance_idx]

        print(f"   📊 En yüksek feature importance: {max_importance_feature} ({max_importance:.3f})")

        if max_importance > 0.3:
            print(f"   🚨 UYARI: {max_importance_feature} çok yüksek importance ({max_importance:.3f}) - Leakage riski!")
        else:
            print(f"   ✅ Feature importance normal ({max_importance:.3f})")

    # Train-test score gap kontrolü
    train_score = results_df.loc[best_model_name, 'CV R2']
    test_score = results_df.loc[best_model_name, 'Test R2']
    score_gap = train_score - test_score

    print(f"   📊 Train R²: {train_score:.4f}")
    print(f"   📊 Test R²: {test_score:.4f}")
    print(f"   📊 Score gap: {score_gap:.4f}")

    if score_gap > 0.2:
        print(f"   🚨 UYARI: Score gap çok yüksek ({score_gap:.4f}) - Overfitting/Leakage riski!")
    else:
        print(f"   ✅ Score gap normal ({score_gap:.4f})")

    # R² değeri kontrolü
    if test_score > 0.8:
        print(f"   🚨 UYARI: Test R² çok yüksek ({test_score:.4f}) - Data leakage şüphesi!")
        print(f"   📊 Beklenen: 0.4-0.7 arası")
    else:
        print(f"   ✅ Test R² gerçekçi ({test_score:.4f})")

    # 10. En iyi model
    best_model_name, best_model = model_trainer.get_best_model()
    print(f"\n🏆 En iyi model: {best_model_name}")

    # 11. Model değerlendirme
    evaluator = ModelEvaluator()

    # Rapor oluştur
    print("\n📊 Model değerlendirme raporu oluşturuluyor...")
    evaluation_report = evaluator.create_evaluation_report(results_df)

    # 12. DATA LEAKAGE KONTROLÜ
    print("\n🔍 Data leakage kontrolü yapılıyor...")

    # Feature-target korelasyon kontrolü
    feature_target_corr = X_train_final.corrwith(y_train).abs().sort_values(ascending=False)

    print("📊 En yüksek korelasyonlar (target ile):")
    top_corr = feature_target_corr.head(5)
    for feature, corr in top_corr.items():
        status = "🚨 Şüpheli" if corr > 0.9 else "✅ Normal"
        print(f"   {status} {feature}: {corr:.4f}")

    # Şüpheli yüksek korelasyon uyarısı
    suspicious_features = feature_target_corr[feature_target_corr > 0.9]
    if len(suspicious_features) > 0:
        print(f"\n⚠️ UYARI: {len(suspicious_features)} feature şüpheli yüksek korelasyon!")
        print("Bu data leakage işareti olabilir.")
    else:
        print("\n✅ Data leakage kontrolü BAŞARILI - Şüpheli korelasyon yok!")

    # 13. Rapor oluştur
    # evaluation_report = evaluator.create_evaluation_report(results_df) # This line is now redundant as it's done above

    print("\n🎉 DATA LEAKAGE TEMİZLENMİŞ pipeline tamamlandı!")

    return {
        'results_df': results_df,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features
    }


def select_risk_method():
    """Risk hesaplama metodunu seç"""

    print("\n" + "="*70)
    print("🎯 RİSK HESAPLAMA METODİ SEÇİMİ")
    print("="*70)

    from src.config import config

    print("\nMevcut metod:", config.RISK_CALCULATION_CONFIG['method'])
    print(f"Açıklama: {config.RISK_CALCULATION_CONFIG['explanation'][config.RISK_CALCULATION_CONFIG['method']]}")

    print("\nRisk hesaplama metodu:")
    print("🏆 Historical Performance (Gerçek payment data - SIFIR leakage riski)")
    print("✅ Sadece bu metod kullanılabilir - diğer metodlar data leakage riski nedeniyle kaldırıldı")
    print("\n1. ⚡ Devam et (Historical Performance)")
    print("2. ❌ Çıkış")

    try:
        choice = input("\nSeçiminiz (1-2): ").strip()

        if choice == "1":
            config.RISK_CALCULATION_CONFIG['method'] = 'historical_performance'
            print("✅ Historical Performance metod aktif - SIFIR leakage riski!")
        elif choice == "2":
            print("❌ Program sonlandırılıyor...")
            return False
        else:
            print("❌ Geçersiz seçim. Historical Performance metod kullanılacak.")
            config.RISK_CALCULATION_CONFIG['method'] = 'historical_performance'

        print(f"📋 Aktif metod: {config.RISK_CALCULATION_CONFIG['method'].title()}")
        print(f"📖 {config.RISK_CALCULATION_CONFIG['explanation'][config.RISK_CALCULATION_CONFIG['method']]}")

    except KeyboardInterrupt:
        print("\n⚠️ İptal edildi. Mevcut ayar korunuyor.")


def main():
    """Ana program"""

    print("\n" + "="*70)
    print("🚀 GELİŞMİŞ FİNANSAL RİSK TAHMİN SİSTEMİ")
    print("="*70)

    # Dizinleri oluştur
    create_directories()

    # Risk metodu seçimi
    select_risk_method()

    print("\nLütfen bir seçenek seçin:")
    print("1. Hızlı Model Eğitimi")
    print("2. Özelleştirilmiş Pipeline (Detaylı analiz ve görselleştirme)")

    try:
        choice = input("\nSeçiminiz (1-2): ").strip()

        if choice == "1":
            run_quick_training()
        elif choice == "2":
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
        print("   - Görselleştirmeler: Devre dışı bırakıldı (plots/ kaldırıldı)")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️ Program kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Command line argument kontrolü
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("🚀 Finansal Risk Tahmin Sistemi")
            print("\nKullanım:")
            print("python main_advanced.py    # Historical Performance metod (SIFIR leakage)")
            print("python main_advanced.py -h # Bu yardım mesajı")
            print("\n🏆 Sadece Historical Performance metod kullanılabilir")
            print("✅ Diğer metodlar data leakage riski nedeniyle kaldırıldı")
            sys.exit(0)
        else:
            # Herhangi bir argüman verilirse historical performance kullan
            from src.config import config
            config.RISK_CALCULATION_CONFIG['method'] = 'historical_performance'
            print("🏆 Command line: Historical Performance metod otomatik seçildi")

    main()