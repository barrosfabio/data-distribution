from distribution.config.global_config import GlobalConfig
import numpy as np
import os
import joblib
from distribution.classification.classification_algorithm import ClassificationAlgorithm
from distribution.resampling.resampling_algorithm import ResamplingAlgorithm
from sklearn.metrics import classification_report
from distribution.config.global_config import GlobalConfig
from distribution.resampling.resampling_constants import NONE

import math


CLASS_SEPARATOR = '/'
model_path = ''

def create_combinations(path, separator, combinations):

    for i in range(1,(len(path)+1)):
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

    return combinations

models_path = '/models'

def save_model(data_class, model):
    global_config = GlobalConfig.instance()

    if data_class != 'R':
        file_name = data_class.replace('/','_')
    else:
        file_name = data_class

    dir_path = model_path +'/fold_'+ str(global_config.kfold) +'/'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    file_path = dir_path + file_name

    joblib.dump(model, file_path)

def load_model(data_class):
    global_config = GlobalConfig.instance()

    if data_class != 'R':
        file_name = data_class.replace('/','_')
    else:
        file_name = data_class

    dir_path = model_path +'/fold_'+ global_config.kfold +'/'

    file_path = dir_path + file_name

    model = joblib.load(file_path)

    return model