'''
This code using a classification method which called Logistic Regression to make predictions on whether there will
be an accident.
For example, give the model of the vehicle, the name of the street and the time, this model will give an output 1 or 0,
where 1 means there will be an accident and 0 means the driver is safe.
The final accuracy of this model is about 96%.
'''

import csv
from sklearn.linear_model import LogisticRegression
import random
import numpy as np

# load the dataset and shuffle it
file_name = 'collision_reports_processed.csv'
data = []
with open(file_name) as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)
print(data[0])
np.random.shuffle(data)

# create veh-model set, address set and time set
dataset = []
veh_set = set()
address_set = set()
time_set = set()
address_acc = {}
veh_model_acc = {}
time_acc = {}
for d in data:
    veh = d['veh_make']
    address = d['street_name_primary']
    time = d['hour_minute'][:2]
    veh_set.add(veh)
    address_set.add(address)
    time_set.add(time)
    dataset.append((veh, address, time))
    # calculate the number of accident for each time, address, vehicle model
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

# sort them by value
veh_model_acc = sorted(veh_model_acc.items(), key=lambda x: x[1], reverse=True)
time_acc = sorted(time_acc.items())
address_acc = sorted(address_acc.items(), key=lambda x: x[1], reverse=True)

# split train_set and test_set by 3:1
split_point = int(0.75 * len(dataset))
train_set = dataset[:split_point]
test_set = dataset[split_point:]

# add negative sample to the original train_set and test_set
new_train_set = []
new_test_set = []
for veh, address, time in train_set:
    new_train_set.append((veh, address, time, 1))
for veh, address, time in test_set:
    new_test_set.append((veh, address, time, 1))
for index, (veh, address, time) in enumerate(train_set):
    random.seed(index * 33)
    new_veh = random.choice(list(veh_set))
    new_address = random.choice(list(address_set))
    new_time = random.choice(list(time_set))
    new_train_set.append((new_veh, new_address, new_time, 0))
for index, (veh, address, time) in enumerate(test_set):
    random.seed(index * 3333)
    new_veh = random.choice(list(veh_set))
    new_address = random.choice(list(address_set))
    new_time = random.choice(list(time_set))
    new_test_set.append((new_veh, new_address, new_time, 0))

# shuffle the train_set and test_set
np.random.shuffle(new_train_set)
np.random.shuffle(new_test_set)


# return the feature for each data, which means return one-hot-encoding for veh-model, address and time
def feature(d, veh_count, address_count):
    veh_one_hot = [0 for i in range(veh_count + 1)]
    time_one_hot = [0 for i in range(24)]
    address_one_hot = [0 for i in range(address_count + 1)]
    veh = d[0]
    address = d[1]
    time = d[2]
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


# return the accuracy of the model
def accuracy(y_real, y_pred):
    count = 0
    for i, j in zip(y_real, y_pred):
        if i == j:
            count += 1
    return count / len(y_real)


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
x_train = [feature(d, veh_count, address_count) for d in new_train_set]
y_train = [d[3] for d in new_train_set]
x_test = [feature(d, veh_count, address_count) for d in new_test_set]
y_test = [d[3] for d in new_test_set]

# create logistic regression (classification) model, fit the model and make predictions
model = LogisticRegression(C=10, fit_intercept=False, max_iter=1000)
model.fit(x_train, y_train)
prediction = model.predict(x_test)

# print the accuracy of the model
print('the accuracy of this model is:')
print(accuracy(prediction, y_test))
# print(prediction[:20])
# print(y_test[:20])

import matplotlib.pyplot as plt

x = np.linspace(0, 50, 50)
plt.plot(x, prediction[:50], label='predictions', color='red')
plt.plot(x, y_test[:50], label='real value', color='blue')

plt.title('Classification of whether there will be a collision')
plt.xlabel('samples')
plt.ylabel('whether there will be a collision')
plt.legend()
plt.show()
