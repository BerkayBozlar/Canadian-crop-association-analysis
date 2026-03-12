import pandas as pd
import ast
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# CSV yükle
df = pd.read_csv("./Tablolar/tarim_sepetleri.csv")

# SEPET sütununu güvenle dönüştür
def guvenli_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df["SEPET"] = df["SEPET"].apply(lambda x: guvenli_literal_eval(x) if pd.notnull(x) else [])

# Bölge filtresi (tüm bölgeler için filtreyi kapattım, sonra tekrar açarsın)
df_ab = df.copy()

# Yıl filtreleri
ilk_70yil = df_ab[df_ab["YIL"] <= 1974]
son_10yil = df_ab[df_ab["YIL"] > 1974]

def analiz_et(veri, etiket):
    transactions = veri["SEPET"].tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Döneme göre destek eşiği ayarla
    destek = 0.4 if etiket == "son_10yil" else 0.4

    # Apriori
    start_apriori = time.time()
    frequent_apriori = apriori(df_encoded, min_support=destek, use_colnames=True)
    rules_apriori = association_rules(frequent_apriori, metric="confidence", min_threshold=0.5)
    end_apriori = time.time()

    # FP-Growth
    start_fpgrowth = time.time()
    frequent_fpgrowth = fpgrowth(df_encoded, min_support=destek, use_colnames=True)
    rules_fpgrowth = association_rules(frequent_fpgrowth, metric="confidence", min_threshold=0.5)
    end_fpgrowth = time.time()

    # Kural filtreleme
    rules_apriori_filtered = rules_apriori[
        (rules_apriori["antecedents"].apply(lambda x: len(x) == 1)) &
        (rules_apriori["consequents"].apply(lambda x: len(x) == 1))
    ]
    rules_fpgrowth_filtered = rules_fpgrowth[
        (rules_fpgrowth["antecedents"].apply(lambda x: len(x) == 1)) &
        (rules_fpgrowth["consequents"].apply(lambda x: len(x) == 1))
    ]

    top_apriori = rules_apriori_filtered.sort_values("lift", ascending=False).head(10)
    top_fpgrowth = rules_fpgrowth_filtered.sort_values("lift", ascending=False).head(10)

    top_apriori["antecedent"] = top_apriori["antecedents"].apply(lambda x: list(x)[0])
    top_apriori["consequent"] = top_apriori["consequents"].apply(lambda x: list(x)[0])
    top_fpgrowth["antecedent"] = top_fpgrowth["antecedents"].apply(lambda x: list(x)[0])
    top_fpgrowth["consequent"] = top_fpgrowth["consequents"].apply(lambda x: list(x)[0])

    top_apriori_out = top_apriori[["antecedent", "consequent", "support", "confidence", "lift"]]
    top_fpgrowth_out = top_fpgrowth[["antecedent", "consequent", "support", "confidence", "lift"]]

    top_apriori_out[["support", "confidence", "lift"]] = top_apriori_out[["support", "confidence", "lift"]].round(2)
    top_fpgrowth_out[["support", "confidence", "lift"]] = top_fpgrowth_out[["support", "confidence", "lift"]].round(2)

    top_apriori_out.to_csv(f"{etiket}_apriori.csv", index=False)
    top_fpgrowth_out.to_csv(f"{etiket}_fpgrowth.csv", index=False)

    print(f"\n--- {etiket.upper()} DÖNEMİ ---")
    print(f"Apriori süresi: {round(end_apriori - start_apriori, 4)} saniye")
    print(f"FP-Growth süresi: {round(end_fpgrowth - start_fpgrowth, 4)} saniye")


analiz_et(ilk_70yil, "ilk_70yil")

analiz_et(son_10yil, "son_10yil")