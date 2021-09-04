import csv
import matplotlib

matplotlib.use('Agg')
import os
import pandas as pd
from distribution.config.global_config import GlobalConfig
from datetime import datetime

"""
    This method creates the directories to store the results of the experiments
"""


def create_result_directories(result_path, classifier_type, file_name, baseline=False):
    global_config = GlobalConfig.instance()

    timestamp = datetime.now()
    timestamp_string = str(timestamp.day) + '_' + str(timestamp.month) + '_' + str(timestamp.hour) + '_' + str(
        timestamp.minute) + '_' + str(global_config.random_seed)

    if baseline:
        result_path = result_path + file_name + '_baseline_' + classifier_type + '_' + timestamp_string
    else:
        result_path = result_path + file_name + '_' + classifier_type + '_' + timestamp_string

    if not os.path.isdir(result_path):
        print('Created directory {}'.format(result_path))
        os.mkdir(result_path)

    # List of directories and sub-directories where the results will be saved
    # This is the basic list
    directory_list = {'resampling': result_path + '/resampling/'}

    for key, value in directory_list.items():
        if not os.path.isdir(value):
            print('Created directory {}'.format(value))
            os.mkdir(value)

    global_config = GlobalConfig.instance()
    global_config.set_directory_configuration(directory_list)


"""
    Transforms a single result into a dataframe and saves it in csv
"""


def transform_to_csv(result, file_name):
    global_config = GlobalConfig.instance()
    columns = result.columns

    path = global_config.directory_list['overall_results']

    file_name = path + file_name

    result_data_frame = pd.DataFrame(columns=columns)

    result_data_frame = result_data_frame.append(result, ignore_index=True)

    write_csv(file_name, result_data_frame)


def transform_multiple_to_csv(result_list, file_name):
    global_config = GlobalConfig.instance()
    columns = result_list[0].index

    path = global_config.directory_list['overall_results']

    if not os.path.isdir(path):
        os.mkdir(path)

    file_name = path + file_name

    result_data_frame = pd.DataFrame(columns=columns)

    for row in result_list:
        result_data_frame = result_data_frame.append(row, ignore_index=True)

    write_csv(file_name, result_data_frame)

def transform_multiple_dict_to_csv(path, result_list, file_name):
    columns = result_list[0].keys()

    if not os.path.isdir(path):
        os.mkdir(path)

    file_name = path + file_name

    result_data_frame = pd.DataFrame(columns=columns)

    for row in result_list:
        result_data_frame = result_data_frame.append(row, ignore_index=True)

    write_csv(file_name, result_data_frame)


"""
    This method writes a data_frame to CSV
"""


def write_csv(file_name, data_frame):
    csv_file_path = file_name + '.csv'
    header = list(data_frame.columns.values)

    print('Saving file to path: {}'.format(csv_file_path))

    with open(csv_file_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL, dialect='excel')

        # Write the header of the table
        filewriter.writerow(header)

        # Write all the other rows
        for index, row in data_frame.iterrows():
            filewriter.writerow(row)


# Saves the class distribution before and after resampling
def save_class_distribution(before_resample, after_resample, class_name, resampler):
    global_config = GlobalConfig.instance()
    data_dist_path = global_config.directory_list['data_distribution']
    data_dist_path = data_dist_path + '/' + 'fold_' + str(global_config.kfold)
    if not os.path.isdir(data_dist_path):
        os.mkdir(data_dist_path)

    class_name = class_name.replace('/', '_')
    before_file_name = data_dist_path + '/before_resample_' + resampler + '_' + class_name
    after_file_name = data_dist_path + '/after_resample_' + resampler + '_' + class_name

    write_csv(before_file_name, before_resample)
    write_csv(after_file_name, after_resample)


def consolidate_result(fold_result_list):
    global_config = GlobalConfig.instance()

    # Calculating the average results
    consolidated_df = pd.concat(fold_result_list)
    consolidated_df = consolidated_df.loc[global_config.metric, :]

    return consolidated_df

def calculate_average_result(fold_result_list):
    global_config = GlobalConfig.instance()

    # Calculating the average results
    concat_df = pd.concat(fold_result_list)
    concat_df = concat_df.loc[global_config.metric, :]
    mean_result = concat_df.mean(axis=0)

    return mean_result
