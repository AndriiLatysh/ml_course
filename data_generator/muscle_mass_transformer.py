import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


muscle_mass_df = pd.read_csv("data/muscle_mass_hours.csv")

for z in range(len(muscle_mass_df)):
    muscle_mass_df.at[z, "training_time"] += random.gauss(0, 0.2)
    muscle_mass_df.at[z, "muscle_mass"] += random.gauss(0, 4)

muscle_mass_df["training_time"] = ((muscle_mass_df["training_time"] * 60) // 5) * 5

muscle_mass_df = muscle_mass_df.astype({"training_time": int, "muscle_mass": int})

muscle_mass_df.to_csv("data/muscle_mass.csv", index=False)

plt.scatter(muscle_mass_df["training_time"], muscle_mass_df["muscle_mass"])

plt.show()
