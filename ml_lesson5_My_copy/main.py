import pandas as pd
import gender_guesser.detector as gender

my_df = pd.read_csv('DataSet_1.csv')
d = gender.Detector()
t = pd.Timestamp.now()

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

def conv_str_to_int(str_val):
    if str_val == 'y':
        return 1
    elif str_val == 'n':
        return 0


def conv_user_rating(usr_rate):
    ratings = {'excellent': 5, 'good': 4, 'ok': 3, 'bad': 2}
    if usr_rate in ratings:
        return ratings[usr_rate]
    else:
        return 1

def convert_age_to_range(age):
    if 6<= age <18:
        return "6-18"
    elif 18 <= age <35:
        return "18-35"
    elif 35 <= age <99:
        return "35-99"
    else:
        return "unknown"


convert_dict = {'user rating': int, 'made purchase': int, 'followed ad': int}
convert_dict_int = {'age': int}


for row in range(len(my_df)):
    if pd.isna(my_df.at[row, "gender"]):
        my_df.at[row, 'gender'] = d.get_gender(my_df.at[row, "first name"])

for row in range(len(my_df)):
    if my_df.at[row, "gender"] == 'female':
        my_df.at[row, "gender"] = 'F'
    elif my_df.at[row, "gender"] == 'male':
        my_df.at[row, "gender"] = 'M'

my_df.insert(1, 'full name', '')
my_df.fillna({'first name': '', 'last name': ''}, inplace=True)
my_df['full name'] = my_df[['first name', 'last name']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
my_df.drop(['first name', 'last name'], axis=1, inplace=True)


my_df.insert(3, 'birthday date', '')
for row in range(len(my_df)):
    my_df.at[row, 'birthday date'] = pd.Timestamp(year=my_df.at[row, 'year of birth'], month=my_df.at[row, 'month of birth'], day=my_df.at[row, 'day of birth'])
my_df.drop(['year of birth', 'month of birth', 'day of birth'], axis=1, inplace=True)

my_df.insert(4, 'age', '')
for row in range(len(my_df)):
    my_df.at[row, 'age'] = (t - (my_df.at[row, 'birthday date'])).days // 365
my_df.drop(['birthday date'], axis=1, inplace=True)


print('\n')
print(my_df.dtypes)
for row in range(len(my_df)):
    my_df.at[row, 'followed ad'] = conv_str_to_int(my_df.at[row, 'followed ad'])
    my_df.at[row, 'made purchase'] = conv_str_to_int(my_df.at[row, 'made purchase'])
    my_df.at[row, 'user rating'] = conv_user_rating(my_df.at[row, 'user rating'])
    my_df.at[row, "age"] = convert_age_to_range(my_df.at[row, "age"])
    if my_df.at[row, 'seen count'] > 1e9:
        my_df.at[row, 'seen count'] = 0

my_df = my_df.astype(convert_dict)
conv_by_color = my_df[['color scheme', 'followed ad', 'made purchase']].groupby('color scheme').mean()
conv_by_gend = my_df[['gender', 'followed ad', 'made purchase']].groupby('gender').mean()

print(conv_by_gend)

print('\n')
print(my_df)
my_df.to_csv('mydf.csv')
