from distribution.data import data_helpers, data_loading
from sklearn.model_selection import StratifiedKFold

from distribution.resampling.resampling_constants import *
from distribution.config.global_config import GlobalConfig
from distribution.result.results_helpers import calculate_average_result,consolidate_result, create_result_directories
from distribution.hierarchy.hierarchical_constants import LCPN_CLASSIFIER, LCN_CLASSIFIER
from distribution.classification.classification_experiment import ClassificationExperiment
from distribution.result.results_helpers import transform_multiple_dict_to_csv


import os
import pandas as pd


path = '../dataset'
result_path = '../final_result/'
folds = 5
classifier_type = LCN_CLASSIFIER
strategy = NONE
resampler = NONE
metric = 'f1-score'
# Initialize seed of the Stratified k-fold split
random_seed = 1


def run_experiment(data_path, filename):
    # Load dataset from CSV
    [data, unique_classes] = data_loading.load_csv_data(data_path)
    [input_data, output_data] = data_helpers.slice_data(data)

    # instantiate singleton and initialize global configurations
    global_config = GlobalConfig.instance()
    global_config.set_random_seed(random_seed)
    global_config.set_metric(metric)
    global_config.set_local_classifier(classifier_type)
    global_config.set_file_name(file_name)

    # For each fold
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)

    # Create the directories to store the experiment results
    create_result_directories(result_path, classifier_type, True)

    resampler_results = []
    global_config.set_resampler_results(resampler_results)

    experiment = ClassificationExperiment(unique_classes, input_data, output_data, classifier_type, strategy, resampler)

    kfold_count = 1

    for train_index, test_index in kfold.split(input_data, output_data):
        print('----------Started fold {} ----------'.format(kfold_count))
        global_config.set_kfold(kfold_count)

        # Calling the classification experiment
        experiment.classification(train_index, test_index)

        # Incrementing the kfold_count
        kfold_count += 1

    transform_multiple_dict_to_csv(global_config.directory_list['resampling'], global_config.resampler_results,
                                       'resamplers_results_' + classifier_type + '_' +strategy)


if __name__ == '__main__':

    directories = os.listdir(path)

    for directory in directories:

        # Sub-directory path
        sub_directory = path + '/' + directory

        # Gathering files
        training_file_list = os.listdir(sub_directory)

        for file in training_file_list:
            # Build file directory path
            file_path = sub_directory + '/' + file

            file_name = file[0:-4]

            run_experiment(file_path, file_name)
