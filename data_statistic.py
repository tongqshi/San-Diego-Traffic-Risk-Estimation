import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from scipy.stats import norm

# read files
collision_data = pd.read_csv('collision_reports_processed.csv')
collision_data = collision_data.fillna('UNKNOWN')
traffic_data = pd.read_csv('traffic_counts_processed.csv')

#check files' colunms or other basic information
print(collision_data.columns)
print(traffic_data.columns)

# calculate the average traffic count of each street
traffic_groupby_street = traffic_data.groupby('street_name')
traffic_average_street = traffic_groupby_street.sum('total_count') / traffic_groupby_street.count()
print(traffic_average_street.columns)

# collision count from different car maker and choose those larger than 20
# question: we dont know the total car number of different car maker, so this is not so persuasive
carmaker_count = defaultdict(int)
for d in collision_data['veh_make']:
    carmaker_count[d] += 1
carmaker_count = sorted(carmaker_count.items(), key = lambda x: x[1], reverse = True)
carmaker_count = {name: count for name, count in carmaker_count[:50]}
print(carmaker_count)
# visualization
plt.figure(figsize=(10, 5))
sns.barplot(x=carmaker_count.keys(), y=carmaker_count.values())
plt.title('collision of different car makers')
plt.xlabel('car makers')
plt.xticks(fontsize=6)
plt.ylabel('collision count')
plt.xticks(rotation=45)
plt.show()

# generate wordcloud of different car maker
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(carmaker_count)
plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# collision count of different month
month_count = collision_data.groupby('month').count()
month_count = month_count['date_time']
print(month_count)
# causal visualization
plt.figure(figsize=(10, 5))
sns.barplot(x=month_count.index, y=month_count.values)
plt.title('collision of different month')
plt.xlabel('month')
plt.ylabel('collision count')
plt.show()

# collision count of different day
day_count = collision_data.groupby('day').count()
day_count = day_count['date_time']
print(day_count)

plt.figure(figsize=(10, 5))
sns.barplot(x=day_count.index, y=day_count.values)
plt.title('collision of different day')
plt.xlabel('day')
plt.ylabel('collision count')
plt.show()

# collision count of different time within a day
hour_count = defaultdict(int)
for t in collision_data['hour_minute']:
    hour_count[t[:2]] += 1

hours = sorted(hour_count.items(), key=lambda x: x[0])
hours = {hour: count for hour, count in hours}

# visualization
plt.figure(figsize=(10, 5))
sns.barplot(x=hours.keys(), y=hours.values())
plt.title('collision of different hours')
plt.xlabel('hours')
plt.ylabel('collision count')
plt.show()

# collision count in minutes
minute_count = collision_data.groupby('hour_minute').count()
minute_count = minute_count['date_time']
minute_count_dict = defaultdict(int)
for index, value in minute_count.items():
    index_minute = (int(index[:2]) * 60 + int(index[-2:]))//5
    minute_count_dict[index_minute] += value

# visualization
plt.figure(figsize=(10, 5))

sns.lineplot(x=minute_count_dict.keys(), y=minute_count_dict.values())
plt.title('collision of different minutes')
plt.xlabel('minutes')
plt.ylabel('collision count')
plt.show()

# collision count of different month and hour to see the influence of seasons
monthour_count = defaultdict(int)
# collision_data[monthour] =
# monthour_count = collision_data.groupby(['month', 'hour_minute']).count()
for d in collision_data.values:
#     print(d)
    index = (d[2] - 1) * 24 + int(d[4][:2])
    monthour_count[index] += 1

monthour_list = list(monthour_count.items())
print(monthour_list.sort())

mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def plot_subplots():
    # generate 3 * 4 subplot meshgrid
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharey=True)

    # plot every subplot
    for i in range(12):
        x = range(24)
        y = [j[1] for j in monthour_list[24 * i: 24 * (i + 1)]]

        # try to fit an gaussian distribution
        #             mean = sum([(m - 24 * i) * n for m, n in monthour_list[24 * i: 24 * (i + 1)]]) / sum(y)
        #             std_dev = sum([(m - 24*i - mean) ** 2 * n for m, n in monthour_list[24 * i: 24 * (i + 1)]]) / sum(y)
        #             xg = np.linspace(5, 23, 100)
        #             yg = norm(xg,mean,std_dev)

        axes[i // 4, i % 4].bar(x, y, color='skyblue', alpha=0.7)
        axes[i // 4, i % 4].set_title(mon[i])

    # avoid overlap
    plt.tight_layout()

    # show graph
    plt.show()

plot_subplots()

# use the same data of last graph, but in line chart version
plt.figure(figsize=(10, 5))
sns.lineplot(x=monthour_count.keys(), y=monthour_count.values())
plt.title('collision of different monthour')
plt.xlabel('monthour')
plt.ylabel('collision count')
plt.xticks(range(17, 12 * 24 + 17, 24), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

# collision count of each street
street_count = collision_data.groupby('street_name_primary').count()
street_count = street_count['date_time']
street_count.sort_values(ascending = False)

# collision count divided by the average traffic count of the street to describe how dangerous the street is
# the collision rate sort of things
dangerous_coefficient = {}
for index, value in street_count.items():
    if index not in traffic_average_street.index: continue
    dangerous_coefficient[index] = value / traffic_average_street.loc[index]['total_count']

dangerous_coefficient = sorted(dangerous_coefficient.items(), key = lambda x: x[1], reverse = True)

# street collision rate rank
print(dangerous_coefficient)

sns.lineplot(x=minute_count_dict.keys(), y=minute_count_dict.values())
plt.title('collision of different minutes')
plt.xlabel('minutes')
plt.ylabel('collision count')
plt.show()

word_frequencies = dict(dangerous_coefficient[:50])

# word cloud of different street
wordcloud = WordCloud(width=800, height=400, background_color = 'white').generate_from_frequencies(word_frequencies)
plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# average injured people for each collision event based on different role
group_by_role = collision_data.groupby('person_role')
role_injured = group_by_role['injured']
role_injured.sum() / role_injured.count()

# average killed people for each collision event based on different role
role_killed = group_by_role['killed']
role_killed.sum() / role_killed.count()

# try to find the most common collision factor
groupby_collisionfactor = collision_data.groupby('charge_desc')
groupby_collisionfactor = groupby_collisionfactor['injured']
groupby_collisionfactor.count().sort_values(ascending = False)

# try to combine the collision factor and the street to see the injure rate and the death rate
collision_data.groupby(['charge_desc', 'street_name_primary']).count().sort_values(by = 'injured', ascending = False)[200:]
print(collision_data.columns)

'''
the next part is try to use probability to see if a specific collision factor happend in a higher frequency
given a certain street and then decide if the street itself has some problems then maybe can give some 
advice about the road improvement, like a more visible traffic signs or rearrange the lines
'''

street_count = collision_data.groupby('street_name_primary').agg(total_incidents = ('date_time', 'count'))
street_count['probability'] = street_count['total_incidents'] / len(collision_data)
street_count = street_count.sort_values(by='probability', ascending=False)
desc_count = collision_data.groupby('charge_desc').agg(total_incidents = ('date_time', 'count'))
print(street_count)

desc_count['probability'] = desc_count['total_incidents'] / len(collision_data)

desc_count = desc_count.sort_values(by='probability', ascending=False)

street_desc_count = collision_data.groupby(['street_name_primary', 'charge_desc']).agg(
    total_incidents=('date_time', 'count'))
street_desc_count['probability'] = street_desc_count['total_incidents'] / len(collision_data)
street_desc_count = street_desc_count.sort_values(by='probability', ascending=False)
print(street_desc_count)

street_problem = defaultdict(set)
for index, row_series in street_desc_count.iterrows():
    street = index[0]
    desc = index[1]
    if row_series['probability'] / street_count.loc[street]['probability'] > 10 * desc_count.loc[desc]['probability']:
        street_problem[street].add(desc)
# the collision factor has an higher probability to happend in the specific street, stored in a dict
print(street_problem)