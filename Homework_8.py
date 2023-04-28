import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('coffee_ratings.csv')
# pd.set_option('display.max_columns', None)

"""1. Which countries are the largest exporters of coffee?"""
# pd.set_option('display.max_rows', None)

def convert_lb_to_kg(weight):
    if 'kg,lbs' in weight:
        return 0
    if 'kg' in weight:
        return int(weight.strip(' kg'))
    if 'lbs' in weight:
        return int(weight.strip(' lbs')) * 0.453592

df['bag_weight'] = df['bag_weight'].apply(convert_lb_to_kg)

def count_total_weight(df):
    return df.assign(total_weight=df['number_of_bags'] * df['bag_weight'])

new_df = df.groupby('country_of_origin', group_keys=False).apply(count_total_weight)
modified_df = new_df.groupby('country_of_origin').agg(weight_by_country=('total_weight', 'sum')).sort_values(by='weight_by_country', ascending=False)

labels = modified_df.head(7).index
plt.bar(labels, modified_df.head(7)['weight_by_country'])
plt.show()

"""2. What correlations between different indicators of coffee assessment?"""
plt.subplots(figsize=(20, 7))

def create_plot(df_arg, key):
    z = np.polyfit(df_arg.index, df_arg[key], 1)
    p = np.poly1d(z)
    plt.plot(df_arg.index, p(df_arg.index), label=[f'{key}_Trend'])

create_plot(df, 'flavor')
create_plot(df, 'aroma')
create_plot(df, 'aftertaste')
create_plot(df, 'body')
create_plot(df, 'balance')
create_plot(df, 'uniformity')
create_plot(df, 'acidity')
create_plot(df, 'sweetness')
plt.legend()
plt.show()
"""It seems like there are strong correlations between sweetness and uniformity grades,
between flavor and balance grades and also between acidity, body and aroma grades.
But we can say that there is some correlation between all grades."""


"""3. How (if) color affects on common sort of coffee?"""
# pd.set_option('display.max_rows', None)
arabica_green = len(df[(df['species'] == 'Arabica') & (df['color'] == 'Green')])
# print(arabica_green)
arabica_bluegreen = len(df[(df['species'] == 'Arabica') & ((df['color'] == 'Bluish-Green') | (df['color'] == 'Blue-Green'))])
# print(arabica_bluegreen)
robusta_green = len(df[(df['species'] == 'Robusta') & (df['color'] == 'Green')])
# print(robusta_green)
robusta_bluegreen = len(df[(df['species'] == 'Robusta') & ((df['color'] == 'Bluish-Green') | (df['color'] == 'Blue-Green'))])
# print(robusta_bluegreen)

X = ['Green', 'Blue-Green']
arabica = [arabica_green / (arabica_bluegreen + arabica_green) * 100,  arabica_bluegreen / (arabica_bluegreen + arabica_green) * 100]
robusta = [robusta_green / (robusta_bluegreen + robusta_green) * 100, robusta_bluegreen / (robusta_bluegreen + robusta_green) * 100]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, arabica, 0.4, label='Arabica')
plt.bar(X_axis + 0.2, robusta, 0.4, label='Robusta')

plt.xticks(X_axis, X)
plt.xlabel("Color")
plt.ylabel("Percentage of color")
plt.title("Percentage of color in both species")
plt.legend()
plt.show()
"""There is no obvious influence of color to species of coffee.
Both species have approx. equal percentage of products of both colors."""


"""4. Does country affects on quality of coffee?"""
country_with_rating_df = df.groupby('country_of_origin').agg(mean_rating=('total_cup_points', 'mean')).sort_values(by='mean_rating', ascending=False)
# print(country_with_rating_df)
plt.bar(country_with_rating_df.head(5).index, country_with_rating_df.head(5)['mean_rating'])
plt.bar(country_with_rating_df.tail(5).index, country_with_rating_df.tail(5)['mean_rating'])
plt.xlabel('Top and Less rated countries')
plt.ylabel('Average rating')
plt.show()
"""As far as all country have a little bit different total cup points we can say that there is some 
(maybe non-significant) influence of country to quality of coffee"""


"""5. If there is significant influence of height on quality of coffee?"""
pd.set_option('display.max_rows', None)
sorted_by_rating_df = df.dropna(subset=['altitude_low_meters', 'altitude_mean_meters', 'altitude_high_meters'])\
    .sort_values(by='total_cup_points', ascending=False)
# print(sorted_by_rating_df[['altitude_low_meters', 'altitude_mean_meters', 'altitude_high_meters']])
plt.plot(sorted_by_rating_df['altitude_high_meters'], label='high')
plt.plot(sorted_by_rating_df['altitude_low_meters'], label='low')
plt.plot(sorted_by_rating_df['altitude_mean_meters'], label='mean')
plt.legend()
plt.show()
"""According to the plot, sorting by total_cup_points doesn't influence on altitude distribution in some
specific way. So, we can assume that there is NO correlation between quality of coffee and altitude"""
