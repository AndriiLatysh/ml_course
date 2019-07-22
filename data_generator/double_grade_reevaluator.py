import random
import numpy as np
import pandas as pd


df = pd.read_csv("double_grade_reevaluated.csv")
print(df)
df["qualifies"] = 0
R = 25
S = 60
for z in range(len(df)):
    tg = df.at[z, "technical_grade"] + random.randrange(-5, 6)
    eg = df.at[z, "english_grade"] + random.randrange(-5, 6)
    if tg + eg > 150 or (tg - S) ** 2 + (eg - S) ** 2 <= R ** 2:
        df.at[z, "qualifies"] = 1
print(df)
df.to_csv("double_grade_reevaluated_0.csv", index=False)
