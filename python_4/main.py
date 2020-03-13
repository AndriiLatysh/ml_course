# unique_values = set([1, 2, 2, 3])
# print(unique_values)

capitals = {"Ukraine": "Kyiv", "Ireland": "Dublin"}
capitals["USA"] = "Washington DC"

print(capitals)

capitals.pop("Ireland")

print(capitals)

for key, value in capitals.items():
    print("{} -> {}".format(key, value))

print(capitals.items())
