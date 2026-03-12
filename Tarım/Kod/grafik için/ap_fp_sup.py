import pandas as pd
import ast
import time
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth

# CSV dosyasını oku
df = pd.read_csv("tarim_sepetleri.csv")

# Güvenli liste çevirici
def safe_parse(sepet):
    try:
        if pd.notna(sepet):
            result = ast.literal_eval(sepet)
            return result if isinstance(result, list) else []
        else:
            return []
    except Exception:
        return []

# Tüm bölgeleri al, sepeti liste olarak çevir
df["SEPET"] = df["SEPET"].apply(safe_parse)

# NaN değerleri olan satırları at
df = df[df["SEPET"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

# Tüm bölgelerin birleşik sepet listesi
transactions = df["SEPET"].tolist()

# Sepetleri makine öğrenmesi için encode et
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# --- Apriori ---
start_ap = time.time()
frequent_itemsets_ap = apriori(df_encoded, min_support=0.1, use_colnames=True)
end_ap = time.time()
frequent_itemsets_ap["Algorithm"] = "Apriori"
frequent_itemsets_ap["Support"] = frequent_itemsets_ap["support"].round(2)
ap_time = round(end_ap - start_ap, 2)

# --- FP-Growth ---
start_fp = time.time()
frequent_itemsets_fp = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)
end_fp = time.time()
frequent_itemsets_fp["Algorithm"] = "FP-Growth"
frequent_itemsets_fp["Support"] = frequent_itemsets_fp["support"].round(2)
fp_time = round(end_fp - start_fp, 2)

# --- Grafik: Support değerlerine göre karşılaştırma ---
def plot_support_chart(df_ap, df_fp, filename):
    ap = df_ap.copy()
    fp = df_fp.copy()
    ap["Itemset"] = ap["itemsets"].apply(lambda x: ', '.join(list(x)))
    fp["Itemset"] = fp["itemsets"].apply(lambda x: ', '.join(list(x)))

    # Ortak itemsetleri al (kısa grafik için)
    merged = pd.merge(ap, fp, on="Itemset", suffixes=("_ap", "_fp"))
    merged = merged.sort_values(by="Support_ap", ascending=False).head(10)

    x = range(len(merged))
    plt.figure(figsize=(12,6))
    plt.bar(x, merged["Support_ap"], width=0.4, label="Apriori", align='center')
    plt.bar([i + 0.4 for i in x], merged["Support_fp"], width=0.4, label="FP-Growth", align='center')
    plt.xticks([i + 0.2 for i in x], merged["Itemset"], rotation=45, ha='right')
    plt.xlabel("Ürün Birliktelikleri")
    plt.ylabel("Support")
    plt.title("Apriori vs FP-Growth - Support Değerleri")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Grafiği üret
plot_support_chart(frequent_itemsets_ap, frequent_itemsets_fp, "apriori_vs_fpgrowth_supports.png")

# --- Zamanları yazdır ---
print(f"Apriori çalışma süresi: {ap_time} saniye")
print(f"FP-Growth çalışma süresi: {fp_time} saniye")
