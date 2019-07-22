import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


object_sizes = pd.read_csv("object_sizes.csv")

object_sizes = object_sizes.applymap(lambda v: int(round(v)))
print(object_sizes)

object_sizes.to_csv("object_sizes.csv", index=False)
