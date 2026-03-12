import pandas as pd
from collections import Counter

df = pd.read_csv("./Tablolar/tarim_sepetleri.csv")

df_ab = df[df["BÖLGE"] == "AB"].copy()

df_ab.loc[:, "SEPET"] = df_ab["SEPET"].apply(lambda x: eval(x))

df_old = df_ab[df_ab["YIL"] <= max(df_ab["YIL"]) - 10]  # İlk 70 yıl
df_recent = df_ab[df_ab["YIL"] > max(df_ab["YIL"]) - 10]  # Son 10 yıl

old_products = [item for sublist in df_old["SEPET"] for item in sublist]
recent_products = [item for sublist in df_recent["SEPET"] for item in sublist]

old_counts = Counter(old_products)
recent_counts = Counter(recent_products)

print("--- Geleneksel vs. Günümüz Ürün Sıklığı ---")
for product in set(old_counts.keys()).union(set(recent_counts.keys())):
    old_freq = old_counts[product]
    recent_freq = recent_counts[product]
    comparison_value = round(old_freq / 7, 2) if old_freq > 0 else 0  
    print(f"{product} = {old_freq} / 7 = {comparison_value}  |  {recent_freq}")

