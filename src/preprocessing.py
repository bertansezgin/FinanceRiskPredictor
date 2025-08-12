import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Tamamen boş olan sütunları sil
    df = df.dropna(axis = 1, how = 'all' )

    # Gerekirse 'Guid', 'UpdateSystemDate_y.1' gibi sistemsel kolonları da kaldır
    system_cols = [col for col in df.columns if 'Guid' in col or 'SystemDate' in col or 'UserName' in col or 'HostName' in col]
    df = df.drop(columns=system_cols, errors='ignore')

    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Güvenli eksik doldurma
    df['InstallmentCount'] = df['InstallmentCount'].fillna(1)
    df['RemainingPrincipalAmount'] = df['RemainingPrincipalAmount'].fillna(0)
    df['AmountTL'] = df['AmountTL'].fillna(0)
    df['PrincipalAmount'] = df['PrincipalAmount'].fillna(1)  # 0'a bölme için

    # 2. Ortalama ödeme (ne kadar taksite bölünmüş)
    df['OrtalamaOdeme'] = df['AmountTL'] / df['InstallmentCount']

    # 3. Kalan borç oranı
    df['KalanOran'] = df['RemainingPrincipalAmount'] / df['PrincipalAmount']
    df['KalanOran'] = df['KalanOran'].clip(0, 1)

    # 4. Eksik ödeme oranı (eksik ödeme = planlanan - gerçekleşen)
    df['EksikOdemeOrani'] = 1 - (df['AmountTL'] / df['PrincipalAmount'])
    df['EksikOdemeOrani'] = df['EksikOdemeOrani'].clip(0, 1)

    # 5. Ödeme yapılmamış mı (AmountTL = 0)
    df['OdenmediMi'] = (df['AmountTL'] == 0).astype(float)

    # 6. Gecikme günü (pozitifse gecikmiş)
    if 'TranDate' in df.columns and 'MaturityDate' in df.columns:
        df['TranDate'] = pd.to_datetime(df['TranDate'], errors='coerce')
        df['MaturityDate'] = pd.to_datetime(df['MaturityDate'], errors='coerce')

        df['OverdueDays'] = (df['TranDate'] - df['MaturityDate']).dt.days
        df['OverdueDays'] = df['OverdueDays'].apply(lambda x: max(x, 0) if pd.notnull(x) else 0)
        df['VaktindenOnceOdeme'] = (df['TranDate'] < df['MaturityDate']).astype(int)
    else:
        df['OverdueDays'] = 0
        df['VaktindenOnceOdeme'] = 0

    return df

