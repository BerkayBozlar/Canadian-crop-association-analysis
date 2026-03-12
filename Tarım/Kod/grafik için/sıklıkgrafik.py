import pandas as pd
import matplotlib.pyplot as plt

# Veriyi oku
df = pd.read_csv("tarim_sepetleri.csv")

# Sepetleri liste haline çevir
df["SEPET"] = df["SEPET"].apply(eval)

# Geçerli 12 eyalet kodunu al (en çok tekrar eden ilk 12 değer)
gecerli_eyaletler = df["BÖLGE"].value_counts().head(12).index.tolist()

# Sadece bu eyaletlerdeki verileri al
df = df[df["BÖLGE"].isin(gecerli_eyaletler)]

# "Barley" geçen satırları al
barley_df = df[df["SEPET"].apply(lambda x: "Oats" in x)]

# Her eyalette "Barley" kaç kez geçmiş?
barley_counts = barley_df["BÖLGE"].value_counts().reindex(gecerli_eyaletler, fill_value=0)

# Sayısal olarak yazdır
print("Beans, all dry (white and coloured) Ürününün Eyaletlere Göre Geçme Sayısı:")
print(barley_counts)

# Grafik oluştur
plt.figure(figsize=(10, 6))
barley_counts.plot(kind="bar", color="sandybrown", edgecolor="black")
plt.title("Oats Ürününün 12 Eyalette Geçme Sıklığı")
plt.xlabel("Eyalet Kodu")
plt.ylabel("Geçme Sayısı")
plt.ylim(50, 80)  # Y ekseni 70 ile 90 arasında olacak şekilde sınırlandırıldı
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

