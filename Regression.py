'''
This code using a regression method which called Linear Regression to make predictions on how much people will be injured
by the accident.
For example, give the model of the vehicle, the name of the street and the time, this model will give a float output,
where the number means how much people will be injured by this accident.
The final MSE of this model is about 0.85-1.
'''

import csv
from sklearn.linear_model import Ridge
import numpy as np

# load the dataset and split it into train_set and test_set
file_name = 'collision_reports_processed.csv'
dataset = []
with open(file_name) as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset.append(row)
np.random.shuffle(dataset)
split_point = int(0.75 * len(dataset))
data_train = dataset[:split_point]
data_test = dataset[split_point:]
print(data_train[0])

# calculate the number of accident for each time, address, vehicle model and sort them by value
address_acc = {}
veh_model_acc = {}
time_acc = {}
for d in dataset:
    veh = d['veh_make']
    address = d['street_name_primary']
    time = d['hour_minute'][:2]
    if veh in veh_model_acc:
        veh_model_acc[veh] += 1
    else:
        veh_model_acc[veh] = 1
    if time in time_acc:
        time_acc[time] += 1
    else:
        time_acc[time] = 1
    if address in address_acc:
        address_acc[address] += 1
    else:
        address_acc[address] = 1
veh_model_acc = sorted(veh_model_acc.items(), key=lambda x: x[1], reverse=True)
time_acc = sorted(time_acc.items())
address_acc = sorted(address_acc.items(), key=lambda x: x[1], reverse=True)


# calculate the MSE
def MSE(y_real, y_pred):
    sum = 0
    for i, j in zip(y_real, y_pred):
        sum += (i - j) ** 2
    return sum / len(y_real)


# return the feature for each data, which means return one-hot-encoding for veh-model, address and time
def feature(d, veh_count, address_count):
    veh_one_hot = [0 for i in range(veh_count + 1)]
    time_one_hot = [0 for i in range(24)]
    address_one_hot = [0 for i in range(address_count + 1)]
    veh = d['veh_make']
    address = d['street_name_primary']
    time = d['hour_minute'][:2]
    for index, (veh_model, veh_count) in enumerate(veh_model_acc):
        if veh == veh_model:
            if veh_count > 40:
                veh_one_hot[index] = 1
            else:
                veh_one_hot[-1] = 1
    for index, (address_name, address_count) in enumerate(address_acc):
        if address == address_name:
            if address_count > 200:
                address_one_hot[index] = 1
            else:
                address_one_hot[-1] = 1
    time_one_hot[int(time)] = 1
    return [1] + veh_one_hot + time_one_hot + address_one_hot


# determine how much one-hot-encoding space we should use
veh_count = 0
address_count = 0
for veh, count in veh_model_acc:
    if count > 40:
        veh_count += 1
for address, count in address_acc:
    if count > 200:
        address_count += 1

# create x_train, x_test, y_train and y_test
x_train = [feature(d, veh_count, address_count) for d in data_train]
y_train = [int(d['injured']) for d in data_train]
x_test = [feature(d, veh_count, address_count) for d in data_test]
y_test = [int(d['injured']) for d in data_test]

# create linear regression model using Ridge, fit the model and make predictions
model = Ridge(alpha=10, fit_intercept=False)
model.fit(x_train, y_train)
prediction = model.predict(x_test)

# print the MSE of the model
print('the MSE of this model is:')
print(MSE(y_test, prediction))

import matplotlib.pyplot as plt

x = np.linspace(0, 100, 100)
plt.plot(x, prediction[:100], label='prediction', color='red')
plt.plot(x, y_test[:100], label='true value', color='blue')

plt.title('Regression of how much people will be injured in the collision')
plt.xlabel('samples')
plt.ylabel('number of injured')
plt.legend()
plt.show()
