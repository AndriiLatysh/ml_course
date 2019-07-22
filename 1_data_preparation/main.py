import pandas as pd
import colour
import copy


def set_printing_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)


def reorder_columns(df):
    columns_list = list(df.columns.values)
    columns_list.insert(columns_list.index("gender"), "birthday")
    columns_list.pop()
    return df[columns_list]


def convert_text_to_binary(text):
    if text == "y":
        return 1
    elif text == "n":
        return 0


def convert_rating_to_grade(rating):
    rating_mapping = {"excellent": 5, "good": 4, "ok": 3, "bad": 2}
    if rating in rating_mapping:
        return rating_mapping[rating]
    else:
        return 1


def convert_age_to_range(age):
    if 6 <= age < 18:
        return "6-18"
    elif 18 <= age < 35:
        return "18-35"
    elif 35 <= age < 99:
        return "35-99"
    else:
        return None


def normalize_column(column):
    column_copy = column.copy()
    min_value = min(column_copy)
    max_value = max(column_copy)
    range_of_values = max_value - min_value
    for z in range(len(column_copy)):
        column_copy[z] = round((column_copy[z] - min_value) / range_of_values, 2)
    return column_copy


set_printing_options()

conversion_df = pd.read_csv("DataSet_1.csv")

print(conversion_df)

conversion_df["last name"].fillna("", inplace=True)

conversion_df.at[8, "gender"] = "F"
conversion_df.at[14, "gender"] = "M"
conversion_df.at[29, "gender"] = "F"
conversion_df.at[37, "gender"] = "M"

for z in range(len(conversion_df)):
    if conversion_df.at[z, "seen count"] > 1e9:
        conversion_df.at[z, "seen count"] = 0

conversion_df.insert(conversion_df.columns.get_loc("last name"), "full name", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "full name"] = (
            conversion_df.at[z, "first name"] + " " + conversion_df.at[z, "last name"]).strip()

# conversion_df["full name"] = (conversion_df["first name"] + " " + conversion_df["last name"]).apply(str.strip)

conversion_df.drop(columns=["first name", "last name"], inplace=True)

for z in range(len(conversion_df)):
    conversion_df.at[z, "followed ad"] = convert_text_to_binary(conversion_df.at[z, "followed ad"])
    conversion_df.at[z, "made purchase"] = convert_text_to_binary(conversion_df.at[z, "made purchase"])
    conversion_df.at[z, "user rating"] = convert_rating_to_grade(conversion_df.at[z, "user rating"])
    # conversion_df.at[z, "color scheme"] = colour.Color(conversion_df.at[z, "color scheme"])

# conversion_df["birthday"] = (
#         conversion_df["year of birth"].map(str) + "-" + conversion_df["month of birth"].map(str) + "-" + conversion_df["day of birth"].map(str)).map(
#     pd.Timestamp)
# conversion_df.insert(conversion_df.columns.get_loc("gender"), "birthday",
#                      (conversion_df["year of birth"].map(str) + "-" + conversion_df["month of birth"].map(str) + "-" +
#                       conversion_df["day of birth"].map(str)).map(pd.Timestamp))

conversion_df.insert(conversion_df.columns.get_loc("gender"), "birthday", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "birthday"] = pd.Timestamp(day=conversion_df.at[z, "day of birth"],
                                                   month=conversion_df.at[z, "month of birth"],
                                                   year=conversion_df.at[z, "year of birth"])

conversion_df.drop(columns=["day of birth", "month of birth", "year of birth"], inplace=True)

conversion_df.insert(conversion_df.columns.get_loc("birthday"), "age", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "age"] = (pd.Timestamp.now() - conversion_df.at[z, "birthday"]).days // 365

conversion_df.drop(columns=["birthday"], inplace=True)

conversion_df.insert(conversion_df.columns.get_loc("gender"), "age bucket", None)

for z in range(len(conversion_df)):
    conversion_df.at[z, "age bucket"] = convert_age_to_range(conversion_df.at[z, "age"])

# conversion_df.drop(columns=["age"], inplace=True)
conversion_df = conversion_df.astype(
    {"age": int, "seen count": float, "followed ad": int, "made purchase": int, "user rating": int})
colors_groped = (conversion_df[["color scheme", "followed ad", "made purchase"]]).groupby("color scheme").mean()
# print(conversion_df.dtypes)

conversion_df["seen count"] = normalize_column(conversion_df["seen count"])

print(conversion_df)

print(colors_groped)

conversion_df.to_csv("prepared_dataset_1.csv")
