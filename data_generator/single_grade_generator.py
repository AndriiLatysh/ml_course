import random
import numpy as np
import pandas as pd


N = 40
grades = [random.randrange(30, 91) for z in range(N)]
df = pd.DataFrame({"grade": grades})
df["qualifies"] = np.where(df["grade"] > 60, 1, 0)
for z in range(len(df)):
    df.at[z, "grade"] += random.randrange(-10, 11)
print(df)
df.to_csv("single_grade.csv", index=False)
