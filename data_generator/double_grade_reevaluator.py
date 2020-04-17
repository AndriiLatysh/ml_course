import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_values(qualifies_double_grade_df):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")


df = pd.read_csv("data/double_grade.csv")
# print(df)
df["qualifies"] = 0
R = 30
S = 75
L = 150
for z in range(len(df)):
    tg = df.at[z, "technical_grade"] + random.randrange(-5, 6)
    eg = df.at[z, "english_grade"] + random.randrange(-5, 6)
    if tg + eg > L or (tg - S) ** 2 + (eg - S) ** 2 <= R ** 2:
        df.at[z, "qualifies"] = 1

# print(df)
print(df.groupby(by="qualifies").count())
plot_values(df)

plt.show()

df.to_csv("data/double_grade_reevaluated.csv", index=False)
