import pandas as pd
import matplotlib
matplotlib.use('Agg')
import operator
import numpy as np
from sklearn.model_selection import train_test_split
from distribution.hierarchy.data import Data


CLASS_SEPARATOR = '/'


# Calculates imbalance ratio
def calculate_imbalance_ratio(majority_class_count, minority_class_count):
    imbalance_ratio = majority_class_count / minority_class_count

    return imbalance_ratio

def count_per_class(output_data):
    original_count = np.unique(output_data, return_counts=True)
    classes = original_count[0]
    count = original_count[1]

    for i in range(0, len(classes)):
        print('Class {}, Count {}'.format(classes[i], count[i]))
    print('')

def count_by_class(output_values):
    label_count = np.unique(output_values, return_counts=True)
    key_count_dict = {}
    genres = label_count[0]
    counts = label_count[1]
    count = pd.DataFrame()

    for i in range(0, len(genres)):
        key_count_dict[genres[i]] = counts[i]

    sorted_dict = dict(sorted(key_count_dict.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_dict.items():
        row = {'class': key, 'count': value}
        count = count.append(row, ignore_index=True)

    return count


def count_by_class(local_class, strategy, output_values, count_results_list):
    label_count = np.unique(output_values, return_counts=True)
    key_count_dict = {}
    genres = label_count[0]
    counts = label_count[1]
    count = []

    for i in range(0, len(genres)):
        key_count_dict[genres[i]] = counts[i]

    sorted_dict = dict(sorted(key_count_dict.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_dict.items():
        row = {'local_class': local_class,'class': key, 'count' : value, 'strategy': strategy}
        count_results_list.append(row)

    return count


# Function to split inputs and outputs
def slice_data(dataset):
    # Slicing the input and output dataset
    input_data = dataset.iloc[:, :-1].values
    output_data = dataset.iloc[:, -1].values

    return [input_data, output_data]

def slice_and_split_data_holdout(input_data, output_data, test_percentage):
    print('Original class distribution')
    count_per_class(output_data)
    # Splitting the dataset in training/test using the Holdout technique
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(input_data, output_data,
                                                                              test_size=test_percentage,
                                                                              random_state=1, stratify=output_data)

    print('Class distribution after holdout')
    count_per_class(outputs_train)

    return [Data(inputs_train, outputs_train),Data(inputs_test, outputs_test)]  # Return train and test dataset separately


# Function to transform inputs and outputs into a data_frame
def array_to_data_frame(inputs, outputs):
    new_data_frame = pd.DataFrame(inputs)
    new_data_frame['class'] = outputs

    return new_data_frame


def create_combinations(path, separator, combinations):
    for i in range(1, (len(path) + 1)):
        combination = path[0:i]
        str_join = separator.join(combination)
        combinations.append(str_join)

    return combinations


def get_possible_classes(classes):
    combinations = []

    for i in range(len(classes)):
        possible_class = str(classes[i])
        class_splitted = possible_class.split(CLASS_SEPARATOR)
        combinations = create_combinations(class_splitted, CLASS_SEPARATOR, combinations)

    combinations = np.unique(combinations)
    print("Getting all possible combinations for each label: {}".format(combinations))

    return combinations


