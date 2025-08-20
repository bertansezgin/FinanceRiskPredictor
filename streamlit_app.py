"""
Basit Finansal Risk Tahmin Sistemi
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import io

from src.automl_system import AutoMLPipeline
from src.batch_predict import predict_all, load_artifacts
from src.loader import load_data

# Sayfa yapılandırması
st.set_page_config(
    page_title="Risk Tahmin Sistemi",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Finansal Risk Tahmin Sistemi")
    st.markdown("---")
    
    # Risk hesaplama metodu seçimi
    st.sidebar.header("🎯 Risk Hesaplama Metodu")
    
    from src.config import config
    
    # Current method
    current_method = config.RISK_CALCULATION_CONFIG['method']
    
    # Method selection
    method_options = {
        'deterministic': '🧮 Deterministik (Explainable AI)',
        'stochastic': '🎲 Stokastik (Kompleks Modelleme)'
    }
    
    selected_method = st.sidebar.selectbox(
        "Risk hesaplama yaklaşımı:",
        options=list(method_options.keys()),
        format_func=lambda x: method_options[x],
        index=0 if current_method == 'deterministic' else 1,
        help="Deterministik: İş kuralları tabanlı, açıklanabilir AI\nStokastik: Gerçek dünya karmaşıklığını modelleyen"
    )
    
    # Update config if changed
    if selected_method != current_method:
        config.RISK_CALCULATION_CONFIG['method'] = selected_method
        st.sidebar.success(f"✅ {method_options[selected_method]} seçildi!")
    
    # Show explanation
    explanation = config.RISK_CALCULATION_CONFIG['explanation'][selected_method]
    st.sidebar.info(f"📖 {explanation}")
    
    # Ana veri dosyasının varlığını kontrol et
    data_file = "data/birlesik_risk_verisi.csv"
    if not os.path.exists(data_file):
        st.error(f"❌ Veri dosyası bulunamadı: {data_file}")
        return
    
    # Veri bilgilerini göster
    try:
        df = pd.read_csv(data_file)
        st.success(f"✅ Veri yüklendi: {len(df):,} müşteri kaydı")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Model Eğitimi")
            st.markdown("Sistemi eğitmek için butona tıklayın:")
            
            if st.button("🚀 Model Eğit", type="primary", use_container_width=True):
                with st.spinner("Model eğitiliyor... Lütfen bekleyin (1-2 dakika sürebilir)..."):
                    try:
                        # AutoML pipeline çalıştır
                        automl = AutoMLPipeline()
                        results = automl.run_automl(df)
                        
                        st.success("✅ Model eğitimi tamamlandı!")
                        
                        # Sonuçları göster
                        if 'metrics' in results:
                            metrics = results['metrics']
                            st.markdown("**Model Performansı:**")
                            st.write(f"- R² Score: {metrics.get('test_r2', 0):.3f}")
                            st.write(f"- RMSE: {metrics.get('test_rmse', 0):.3f}")
                            st.write(f"- Model: {results.get('best_model_name', 'Bilinmiyor')}")
                        
                    except Exception as e:
                        st.error(f"❌ Eğitim hatası: {str(e)}")
        
        with col2:
            st.markdown("### 📊 Risk Analizi & İndirme")
            st.markdown("Tüm müşterilerin risk analizini yapın ve Excel olarak indirin:")
            
            # Model varlığını kontrol et
            try:
                artifacts = load_artifacts()
                st.success("✅ Eğitilmiş model bulundu")
                
                if st.button("🔮 Risk Analizi Yap", type="primary", use_container_width=True):
                    with st.spinner("Tüm müşteriler için risk analizi yapılıyor..."):
                        try:
                            # Batch prediction yap
                            predictions_df = predict_all()
                            
                            st.success(f"✅ {len(predictions_df):,} müşteri için analiz tamamlandı!")
                            
                            # Sonuçları göster
                            st.markdown("**Analiz Özeti:**")
                            risk_counts = predictions_df['RiskCategory'].value_counts()
                            for category, count in risk_counts.items():
                                percentage = (count / len(predictions_df)) * 100
                                st.write(f"- {category}: {count:,} müşteri (%{percentage:.1f})")
                            
                            # Örnek sonuçları göster
                            st.markdown("**İlk 10 Sonuç:**")
                            st.dataframe(predictions_df.head(10))
                            
                            # Excel indirme butonu hazırla
                            excel_buffer = io.BytesIO()
                            
                            # Excel dosyası oluştur
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                predictions_df.to_excel(writer, sheet_name='Risk Analizi', index=False)
                                
                                # Özet sayfası ekle
                                summary_df = pd.DataFrame({
                                    'Risk Kategorisi': risk_counts.index,
                                    'Müşteri Sayısı': risk_counts.values,
                                    'Yüzde': [(count / len(predictions_df)) * 100 for count in risk_counts.values]
                                })
                                summary_df.to_excel(writer, sheet_name='Özet', index=False)
                            
                            excel_buffer.seek(0)
                            
                            # İndirme butonu
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            st.download_button(
                                label="📥 Excel Olarak İndir",
                                data=excel_buffer.getvalue(),
                                file_name=f"risk_analizi_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Analiz hatası: {str(e)}")
                
            except Exception as e:
                st.warning("⚠️ Eğitilmiş model bulunamadı. Önce model eğitimi yapın.")
        
        # Veri önizleme
        st.markdown("---")
        st.markdown("### 👀 Veri Önizleme")
        
        with st.expander("Veri detaylarını göster", expanded=False):
            st.markdown(f"**Toplam Kayıt:** {len(df):,}")
            st.markdown(f"**Sütun Sayısı:** {len(df.columns):,}")
            
            # İlk 5 satırı göster
            st.markdown("**İlk 5 kayıt:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Temel istatistikler
            if 'ProjectId' in df.columns:
                unique_projects = df['ProjectId'].nunique()
                st.markdown(f"**Benzersiz Proje Sayısı:** {unique_projects:,}")
        
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")

if __name__ == "__main__":
    main()
