import pandas as pd
import time
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# 🔹 Veriyi oku
df = pd.read_csv("./Tablolar/tarim_sepetleri.csv")

# 🔹 Alberta (AB) bölgesine odaklan
df_ab = df[df["BÖLGE"] == "AB"].copy()

# 🔹 Sepetleri liste formatına çevir
df_ab["SEPET"] = df_ab["SEPET"].apply(lambda x: eval(x))

# 🔹 Item sayısına göre veri setleri oluştur
item_counts = list(range(2, 21, 2))
apriori_times = []
fpgrowth_times = []

for item_count in item_counts:
    dataset = [basket[:item_count] for basket in df_ab["SEPET"]]  # Belirtilen item sayısı kadar kes

    # 🔹 Veriyi dönüştür
    encoder = TransactionEncoder()
    encoded_data = encoder.fit(dataset).transform(dataset)
    df_transformed = pd.DataFrame(encoded_data, columns=encoder.columns_)

    # ⏳ Apriori Çalıştırma Zamanı
    start_time = time.time()
    apriori(df_transformed, min_support=0.2, use_colnames=True)
    apriori_times.append(time.time() - start_time)

    # ⏳ FP-Growth Çalıştırma Zamanı
    start_time = time.time()
    fpgrowth(df_transformed, min_support=0.2, use_colnames=True)
    fpgrowth_times.append(time.time() - start_time)

# 🔹 Grafik Çizimi
plt.figure(figsize=(10, 5))
plt.plot(item_counts, apriori_times, marker='o', linestyle='-', label="Apriori")
plt.plot(item_counts, fpgrowth_times, marker='s', linestyle='-', label="FP-Growth")
plt.xlabel("Item Sayısı")
plt.ylabel("Çalışma Süresi (saniye)")
plt.title("Apriori vs FP-Growth Çalışma Süresi Karşılaştırması")
plt.legend()
plt.grid()
plt.show()
