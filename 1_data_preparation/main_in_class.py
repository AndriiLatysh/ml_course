import pandas as pd


def set_printing_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)


def convert_text_to_binary(text):
    if text == "y":
        return 1
    elif text == "n":
        return 0


def convert_rating_to_grade(rating):
    rating_map = {"excellent": 5, "good": 4, "ok": 3, "bad": 2}
    if rating in rating_map:
        return rating_map[rating]
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
        return "unknown"


set_printing_options()
conversions_df = pd.read_csv("DataSet_1.csv")
conversions_df["last name"].fillna("", inplace=True)

conversions_df.at[8, "gender"] = "F"
conversions_df.at[14, "gender"] = "M"
conversions_df.at[29, "gender"] = "F"
conversions_df.at[37, "gender"] = "M"

for z in range(len(conversions_df)):
    if conversions_df.at[z, "seen count"] > 1e9:
        conversions_df.at[z, "seen count"] = 0

conversions_df.insert(conversions_df.columns.get_loc("last name"), "full name", None)
for z in range(len(conversions_df)):
    conversions_df.at[z, "full name"] = (
            conversions_df.at[z, "first name"] + " " + conversions_df.at[z, "last name"]).strip()
conversions_df.drop(columns=["first name", "last name"], inplace=True)

conversions_df.insert(conversions_df.columns.get_loc("gender"), "birthday", None)
for z in range(len(conversions_df)):
    conversions_df.at[z, "birthday"] = pd.Timestamp(day=conversions_df.at[z, "day of birth"],
                                                    month=conversions_df.at[z, "month of birth"],
                                                    year=conversions_df.at[z, "year of birth"])
conversions_df.drop(columns=["day of birth", "month of birth", "year of birth"], inplace=True)

conversions_df.insert(conversions_df.columns.get_loc("birthday"), "age", None)
for z in range(len(conversions_df)):
    conversions_df.at[z, "age"] = (pd.Timestamp.now() - conversions_df.at[z, "birthday"]).days // 365
conversions_df.drop(columns=["birthday"], inplace=True)

for z in range(len(conversions_df)):
    conversions_df.at[z, "followed ad"] = convert_text_to_binary(conversions_df.at[z, "followed ad"])
    conversions_df.at[z, "made purchase"] = convert_text_to_binary(conversions_df.at[z, "made purchase"])
    conversions_df.at[z, "user rating"] = convert_rating_to_grade(conversions_df.at[z, "user rating"])

for z in range(len(conversions_df)):
    conversions_df.at[z, "age"] = convert_age_to_range(conversions_df.at[z, "age"])

conversions_df = conversions_df.astype(
    {"seen count": float, "followed ad": int, "made purchase": int, "user rating": int})

conversions_by_color = conversions_df[["color scheme", "followed ad", "made purchase"]].groupby("color scheme").mean()

conversions_df.to_csv("PreparedDataSet_1.csv")
