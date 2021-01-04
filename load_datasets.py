import numpy as np
import random
import csv
import math


def load_iris_dataset(train_ratio):
    random.seed(1)
    data = []
    target = []
    num_setosa = 0
    num_versicolor = 0
    num_virginica = 0
    try:
        f = open('datasets/bezdekIris.data', 'r')

        # load data from csv file
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        my_list = list(reader)
        random.shuffle(my_list)
        # separate data from target
        count = 0
        for row in my_list:
            if row != []:
                sub_data = np.empty(4)
                for inc in range(4):
                    sub_data[inc] = row[inc]

                sub_t = np.empty(1)
                if row[4] == 'Iris-setosa':
                    sub_t = 0
                    num_setosa += 1
                elif row[4] == 'Iris-versicolor':
                    sub_t = 1
                    num_versicolor += 1
                elif row[4] == 'Iris-virginica':
                    sub_t = 2
                    num_virginica += 1

                target.append(sub_t)
                data.append(sub_data)
            count += 1
        data = np.array(data).astype('float')

    finally:
        f.close()

    percent_setosa = int(math.floor(num_setosa * train_ratio))
    percent_versicolor = int(math.floor(num_versicolor * train_ratio))
    percent_virginia = int(math.floor(num_virginica * train_ratio))

    # create train / test partitions
    size = len(data)
    train = []
    train_labels = []
    test = []
    test_labels = []
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for c in range(size):
        value = target[c]
        if value == 0 and count_0 < percent_setosa:
            train.append(data[c])
            train_labels.append(target[c])
            count_0 += 1
        elif value == 1 and count_1 < percent_versicolor:
            train.append(data[c])
            train_labels.append(target[c])
            count_1 += 1
        elif value == 2 and count_2 < percent_virginia:
            train.append(data[c])
            train_labels.append(target[c])
            count_2 += 1
        else:
            test.append(data[c])
            test_labels.append(target[c])

    train = np.array(train)
    train_labels = np.array(train_labels)
    test = np.array(test)
    test_labels = np.array(test_labels)

    return train, train_labels, test, test_labels