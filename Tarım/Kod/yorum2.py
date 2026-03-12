import pandas as pd
import ast
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

df = pd.read_csv("./Tablolar/tarim_sepetleri.csv")

df_ab = df[df["BÖLGE"] == "AB"].copy()

df_ab["SEPET"] = df_ab["SEPET"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

transactions = df_ab["SEPET"].tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

start_apriori = time.time()
frequent_apriori = apriori(df_encoded, min_support=0.3, use_colnames=True)
rules_apriori = association_rules(frequent_apriori, metric="confidence", min_threshold=0.5)
end_apriori = time.time()

start_fpgrowth = time.time()
frequent_fpgrowth = fpgrowth(df_encoded, min_support=0.3, use_colnames=True)
rules_fpgrowth = association_rules(frequent_fpgrowth, metric="confidence", min_threshold=0.5)
end_fpgrowth = time.time()

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

top_apriori_out.to_csv("apriori_sonuclar.csv", index=False)
top_fpgrowth_out.to_csv("fpgrowth_sonuclar.csv", index=False)

print(f"Apriori süresi: {round(end_apriori - start_apriori, 4)} saniye")
print(f"FP-Growth süresi: {round(end_fpgrowth - start_fpgrowth,4)}saniye")