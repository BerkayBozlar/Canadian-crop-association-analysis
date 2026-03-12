import pandas as pd

df = pd.read_csv("./Tablolar/farm_production_dataset.csv")

df = df[['REF_DATE', 'GEO', 'Type of crop']]

df.rename(columns={'GEO': 'BÖLGE', 'REF_DATE': 'YIL'}, inplace=True)

grouped = df.groupby(['YIL', 'BÖLGE'])['Type of crop'].apply(list).reset_index()

grouped.rename(columns={'Type of crop': 'SEPET'}, inplace=True)

grouped.to_csv("./Tablolar/tarim_sepetleri.csv", index=False)

