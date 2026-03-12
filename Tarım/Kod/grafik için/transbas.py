import pandas as pd
import time
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pyfpgrowth

# CSV'yi oku
df = pd.read_csv("tarim_sepetleri.csv")

# Sepetleri liste olarak ayarla
df["SEPET"] = df["SEPET"].apply(eval)

# Transaction sayıları (örnekleme aralıkları)
transaction_limits = list(range(100, len(df) + 1, 100))

# Sonuç listeleri
apriori_times = []
fpgrowth_times = []
transaction_counts = []

for limit in transaction_limits:
    subset = df["SEPET"][:limit]
    
    # Boşları temizle
    subset = [sepet for sepet in subset if sepet]

    # TransactionEncoder ile One-hot encode
    te = TransactionEncoder()
    oht = te.fit_transform(subset)
    df_oht = pd.DataFrame(oht, columns=te.columns_)

    # Apriori süresi
    start = time.time()
    apriori(df_oht, min_support=0.2, use_colnames=True)
    end = time.time()
    apriori_times.append(end - start)

    # FP-Growth süresi
    start = time.time()
    min_occurrences = int(len(subset) * 0.2)
    pyfpgrowth.find_frequent_patterns(subset, min_occurrences)
    end = time.time()
    fpgrowth_times.append(end - start)

    transaction_counts.append(limit)
    print(f"{limit} transaction → Apriori: {apriori_times[-1]:.4f}s, FP-Growth: {fpgrowth_times[-1]:.4f}s")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(transaction_counts, apriori_times, marker='o', color='blue', label='Apriori')
plt.xlabel("Transaction Sayısı")
plt.ylabel("Çalışma Süresi (saniye)")
plt.title("Transaction Sayısına Göre Apriori Süresi")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
