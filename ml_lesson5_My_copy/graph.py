import pandas as pd
import matplotlib.pyplot as plt

my_df = pd.read_csv('mydf.csv')

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

# # print(my_df[:1])
print(my_df)

aver_follow = my_df[my_df['followed ad'] == 1].mean()
aver_follow_made = my_df[my_df['followed ad'] == 1]['made purchase'].mean()
# print('\n', 'average from following ad = ', aver_follow)
# print('\n', 'average from following ad made purch = ', aver_follow_made, '\n')

cross = pd.crosstab(my_df['followed ad'], my_df['made purchase'])
# print(cross)
# print('\n')

cross_v = my_df.groupby(['followed ad'])[['made purchase']].mean()
# print(cross_v)
# print('\n')

# # print('\n', my_df.describe())

# df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
# ['Area code'], aggfunc='mean').head(10)

piv = my_df.pivot_table(['made purchase'], ['followed ad'], aggfunc='mean')
# print(piv)

# my_df.plot(x='seen count', y='age', kind='scatter')
# titanic.pivot_table('survived', index='sex', columns='class')
# titanic.pivot_table('survived', ['sex', age], 'class')
# plt.show()

piv_full = my_df.groupby(['gender', 'color scheme'])['seen count'].aggregate('mean').unstack()
piv_full_piv = my_df.pivot_table('seen count', index='gender', columns='color scheme')
piv_full_age = my_df.pivot_table('seen count', ['gender', 'age'], 'color scheme')
# piv_full_age = my_df.pivot_table(my_df['seen count'].astype(int), ['gender', 'age'], 'color scheme')

# print(piv_full)
# print('\n')
print(piv_full_piv)
# # print(my_df.groupby(['color scheme'])[['seen count']]).mean()

conv_by_color = my_df[['color scheme', 'seen count', 'followed ad', 'made purchase']].groupby('color scheme').mean()

# print(conv_by_color)

print('\n', 'PIVOT BY AGE', '\n')
print(piv_full_age)

blue_F_35_99 = my_df[(my_df["gender"] == "F") & (my_df["age"] == "35-99") & (my_df["color scheme"] == "blue")]
print("\n", blue_F_35_99)

red_M_6_18 = my_df[(my_df["gender"] == "M") & (my_df["age"] == "6-18") & (my_df["color scheme"] == "red")]
print("\n", red_M_6_18)

# print(my_df.dtypes)