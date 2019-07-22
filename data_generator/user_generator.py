import random
import numpy as np
import pandas as pd
import uuid


def set_printing_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)


set_printing_options()

N = 60
df = pd.DataFrame(columns=["user_id", "items_purchased", "items_returned", "total_spent", "overall_rating"],
                  index=range(N))
for z in range(0, N // 3):
    df.at[z, "user_id"] = uuid.uuid4()
    df.at[z, "items_purchased"] = random.randint(1, 31)
    if random.random() < 0.8:
        df.at[z, "items_returned"] = 0
    else:
        df.at[z, "items_returned"] = random.randint(1, df.at[z, "items_purchased"] // 10 + 1)
    df.at[z, "total_spent"] = int(
        round(random.gauss(df.at[z, "items_purchased"] * 1000, df.at[z, "items_purchased"] * 1000 // 5)))
    df.at[z, "overall_rating"] = int(round(random.triangular(4, 5, 4.8)))

for z in range(N // 3, 2 * N // 3):
    df.at[z, "user_id"] = uuid.uuid4()
    df.at[z, "items_purchased"] = random.randint(1, 21)
    if random.random() < 0.6:
        df.at[z, "items_returned"] = 0
    else:
        df.at[z, "items_returned"] = random.randint(1, df.at[z, "items_purchased"] // 8 + 1)
    df.at[z, "total_spent"] = int(
        round(random.gauss(df.at[z, "items_purchased"] * 600, df.at[z, "items_purchased"] * 600 // 4)))
    df.at[z, "overall_rating"] = int(round(random.triangular(4, 5, 4.6)))

for z in range(2 * N // 3, N):
    df.at[z, "user_id"] = uuid.uuid4()
    df.at[z, "items_purchased"] = random.randint(1, 11)
    if random.random() < 0:
        df.at[z, "items_returned"] = 0
    else:
        df.at[z, "items_returned"] = random.randint(1, df.at[z, "items_purchased"] // 2 + 1)
    df.at[z, "total_spent"] = int(
        round(random.gauss(df.at[z, "items_purchased"] * 750, df.at[z, "items_purchased"] * 750 // 4)))
    df.at[z, "overall_rating"] = int(round(random.triangular(1, 4, 2.5)))

df = df.sample(frac=1)
print(df)

df.to_csv("users_online_closing_store.csv", index=False)
