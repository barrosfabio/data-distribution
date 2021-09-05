from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN, KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, TomekLinks, EditedNearestNeighbours, \
    NearMiss, AllKNN
from distribution.resampling.handle_adasyn import handle_adasyn
from distribution.resampling.resampling_constants import *
from distribution.hierarchy.hierarchical_constants import NEGATIVE_CLASS
from distribution.resampling.resampling_constants import LOCAL_RESAMPLING, IR_SELECTIVE_RESAMPLING
from distribution.data.data_helpers import calculate_imbalance_ratio, count_by_class, slice_data, count_per_class


import numpy as np
class ResamplingAlgorithm:

    def __init__(self, algorithm_name, strategy, random_state=1,  k_neighbors=3):
        self.algorithm_name = algorithm_name
        self.sampling_strategy = 'auto'
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = -1
        self.resampler = self.instantiate_resampler(algorithm_name)
        self.resampling_strategy = strategy

    # Instantiates a resampling algorithm based on the parameter provided
    def instantiate_resampler(self, algorithm_name):

        smote = SMOTE(sampling_strategy=self.sampling_strategy, k_neighbors=self.k_neighbors, random_state=42, n_jobs=self.n_jobs)

        if algorithm_name == SMOTE_RESAMPLE:
            return smote
        elif algorithm_name == SMOTE_ENN:
            return SMOTEENN(sampling_strategy=self.sampling_strategy, random_state=self.random_state, n_jobs=self.n_jobs, smote=smote)
        elif algorithm_name == SMOTE_TOMEK:
            return SMOTETomek(sampling_strategy=self.sampling_strategy, random_state=self.random_state, n_jobs=self.n_jobs, smote=smote)
        elif algorithm_name == BORDERLINE_SMOTE:
            return BorderlineSMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state, n_jobs=self.n_jobs,
                                   k_neighbors=self.k_neighbors)
        elif algorithm_name == ADASYN_RESAMPLER:
            return ADASYN(sampling_strategy=self.sampling_strategy, random_state=self.random_state, n_jobs=self.n_jobs,
                          n_neighbors=self.k_neighbors)
        elif algorithm_name == RANDOM_OVERSAMPLER:
            return RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        elif algorithm_name == KMEANS_SMOTE:
            return KMeansSMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state, n_jobs=self.n_jobs)
        elif algorithm_name == SVM_SMOTE:
            return SVMSMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state, n_jobs=self.n_jobs,
                            k_neighbors=self.k_neighbors)
        elif algorithm_name == RANDOM_UNDERSAMPLER:
            return RandomUnderSampler(sampling_strategy='majority')
        elif algorithm_name == NEIGHBOUR_CLEANING:
            return NeighbourhoodCleaningRule(sampling_strategy='majority', n_jobs=self.n_jobs)
        elif algorithm_name == TOMEK:
            return TomekLinks(sampling_strategy='majority', n_jobs=self.n_jobs)
        elif algorithm_name == ENN:
            return EditedNearestNeighbours(sampling_strategy='majority', n_jobs=self.n_jobs)
        elif algorithm_name == NEAR_MISS:
            return NearMiss(sampling_strategy='majority', n_jobs=self.n_jobs)
        elif algorithm_name == ALL_KNN:
            return AllKNN(sampling_strategy='majority', n_jobs=self.n_jobs)

    # Executes resampling
    def resample(self, input_data, output_data):

        class_count = len(np.unique(output_data))

        if class_count > 1:
            if self.algorithm_name == ADASYN_RESAMPLER:
                [input_data, output_data] = self.handle_adasyn_algorithm(input_data, output_data)
            else:
                [input_data, output_data] = self.resampler.fit_resample(input_data, output_data)

        return [input_data, output_data]

        # Executes hierarchical resampling for LCN Classifier

    def local_resample_lcn(self, input_data, output_data, class_name):

        before_resample = count_by_class(output_data)
        class_count = len(np.unique(output_data))

        negative_class = before_resample[before_resample['class'] == NEGATIVE_CLASS]
        positive_class = before_resample[before_resample['class'] == class_name]
        imbalance_ratio = calculate_imbalance_ratio(negative_class.iloc[0, 1], positive_class.iloc[0, 1])
        print('Imbalance Ratio Other/Interest Class is {}'.format(imbalance_ratio))

        if self.resampling_strategy != IR_SELECTIVE_RESAMPLING and class_count > 1:

            if self.algorithm_name == ADASYN_RESAMPLER:
                [input_data, output_data] = self.handle_adasyn_algorithm(input_data, output_data)
            else:
                [input_data, output_data] = self.resample(input_data, output_data)

        elif self.resampling_strategy == IR_SELECTIVE_RESAMPLING and class_count > 1:

            if imbalance_ratio > IMBALANCE_RATIO:
                if self.algorithm_name == ADASYN_RESAMPLER:
                    [input_data, output_data] = self.handle_adasyn_algorithm(input_data, output_data)
                else:
                    [input_data, output_data] = self.resample(input_data, output_data)


            else:
                pass
        else:
            print('Resampling not needed. Difference between Negatove and Positive class is: {}'.format
                  (negative_class.iloc[0, 1] - positive_class.iloc[0, 1]))

        return [input_data, output_data]

    def handle_adasyn_algorithm(self, input_data, output_data):
        # Check if it is possible to resample the dataset with ADASYN and check which classes will be resampled
        samples_to_generate = handle_adasyn(input_data, output_data, self.k_neighbors)

        # Instantiate specific ADASYN
        self.resampler = ADASYN(sampling_strategy=samples_to_generate, random_state=42, n_jobs=self.n_jobs,
                                n_neighbors=self.k_neighbors)

        if len(samples_to_generate) > 0:
            [input_data, output_data] = self.resampler.fit_resample(input_data, output_data)
        else:
            print('No resampling is needed for this this distribution')

        return [input_data, output_data]
