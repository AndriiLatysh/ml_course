import random
import numpy as np
import pandas as pd


N = 40
technical_grades = [random.randrange(30, 91) for z in range(N)]
english_grades = [random.randrange(30, 91) for z in range(N)]
df = pd.DataFrame({"technical_grade": technical_grades, "english_grade": english_grades})
df["qualifies"] = np.where(df["technical_grade"] + df["english_grade"] > 120, 1, 0)
print(df)
# for z in range(len(df)):
#     df.at[z, "technical_grade"] += random.randrange(-10, 11)
#     df.at[z, "english_grade"] += random.randrange(-10, 11)
# print(df)
df.to_csv("double_grade_small.csv", index=False)
