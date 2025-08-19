"""
Veri Ön İşleme - Data Leakage Önlenmiş Versiyon
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Veriyi temizle"""
    
    # Tamamen boş olan sütunları sil
    df = df.dropna(axis=1, how='all')
    
    # Sistemsel kolonları kaldır
    system_cols = [col for col in df.columns if any(x in col for x in 
                   ['Guid', 'SystemDate', 'UserName', 'HostName', 
                    'UpdateUser', 'CreateUser', 'UpdateHost', 'CreateHost'])]
    df = df.drop(columns=system_cols, errors='ignore')
    
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temel özellikler oluştur - Data Leakage olmadan
    
    NOT: AmountTL kullanılmıyor çünkü bu tahsilat sonrası bilgi!
    Sadece kredi verilirken bilinen değerler kullanılıyor.
    """
    
    # 1. Güvenli eksik doldurma
    df['InstallmentCount'] = df['InstallmentCount'].fillna(1)
    df['RemainingPrincipalAmount'] = df['RemainingPrincipalAmount'].fillna(0)
    df['PrincipalAmount'] = df['PrincipalAmount'].fillna(1)
    df['FundingAmount'] = df['FundingAmount'].fillna(0)
    
    # 2. Kalan borç oranı (Bu güvenli - başlangıç bilgisi)
    df['KalanOran'] = df['RemainingPrincipalAmount'] / df['PrincipalAmount'].replace(0, 1)
    df['KalanOran'] = df['KalanOran'].clip(0, 1)
    
    # 3. Gecikme hesaplama için alternatif yaklaşımlar
    
    # Yaklaşım 1: Eğer tahsilat yapılmamışsa (ProjectCollectionId boş) ve vade geçmişse
    if 'ProjectCollectionId' in df.columns:
        df['TahsilatYapilmadi'] = df['ProjectCollectionId'].isna().astype(float)
    else:
        df['TahsilatYapilmadi'] = 0
    
    # Yaklaşım 2: Gecikme günleri (tarih bilgilerinden)
    if 'TranDate' in df.columns and 'MaturityDate' in df.columns:
        df['TranDate'] = pd.to_datetime(df['TranDate'], errors='coerce')
        df['MaturityDate'] = pd.to_datetime(df['MaturityDate'], errors='coerce')
        
        # Bugünün tarihini referans al
        today = pd.Timestamp.now()
        
        # Eğer TranDate boşsa ve MaturityDate geçmişse -> gecikme var
        df['OverdueDays'] = 0
        
        # TranDate doluysa: TranDate - MaturityDate
        mask_tran = df['TranDate'].notna()
        df.loc[mask_tran, 'OverdueDays'] = (
            df.loc[mask_tran, 'TranDate'] - df.loc[mask_tran, 'MaturityDate']
        ).dt.days
        
        # TranDate boşsa ve vade geçmişse: today - MaturityDate
        mask_no_tran = df['TranDate'].isna() & (df['MaturityDate'] < today)
        df.loc[mask_no_tran, 'OverdueDays'] = (
            today - df.loc[mask_no_tran, 'MaturityDate']
        ).dt.days
        
        # Negatif değerleri 0 yap
        df['OverdueDays'] = df['OverdueDays'].clip(lower=0)
    else:
        df['OverdueDays'] = 0
    
    # 4. Risk göstergeleri (AmountTL kullanmadan)
    
    # Eksik ödeme tahmini - Kalan oran yüksekse ödeme yapılmamış olabilir
    df['EksikOdemeOrani'] = df['KalanOran']  # Basitleştirilmiş
    
    # Ödenmedi göstergesi - Vade geçmiş ve hala kalan borç varsa
    df['OdenmediMi'] = ((df['OverdueDays'] > 0) & (df['KalanOran'] > 0.5)).astype(float)
    
    # 5. Ek güvenli özellikler
    
    # Kredi büyüklük kategorisi
    df['KrediBuyuklukKategorisi'] = pd.qcut(
        df['PrincipalAmount'], 
        q=[0, 0.25, 0.5, 0.75, 1.0],
        labels=[0, 1, 2, 3],
        duplicates='drop'
    ).astype(float).fillna(0)
    
    # Taksit yükü
    df['TaksitYuku'] = df['PrincipalAmount'] / df['InstallmentCount'].replace(0, 1)
    
    # Fonlama oranı
    df['FonlamaOrani'] = df['FundingAmount'] / df['PrincipalAmount'].replace(0, 1)
    df['FonlamaOrani'] = df['FonlamaOrani'].clip(0, 2)
    
    return df