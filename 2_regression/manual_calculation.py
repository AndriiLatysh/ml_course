import pandas as pd


subscribers_from_ads = pd.read_csv("data/subscribers_from_ads.csv")
x = subscribers_from_ads["promotion_budget"].to_list()
y = subscribers_from_ads["subscribers"].to_list()
k = (len(x) * sum([a * b for a, b in zip(x, y)]) - sum(x) * sum(y)) / (len(x) * sum([a ** 2 for a in x]) - sum(x) ** 2)
print(k)
b = (sum(y) * sum([a ** 2 for a in x]) - sum(x) * sum([a * b for a, b in zip(x, y)])) / (
            len(x) * sum([a ** 2 for a in x]) - sum(x) ** 2)
b1 = (1 / len(x)) * sum(y) - (k / len(x)) * sum(x)
print(b, b1)
