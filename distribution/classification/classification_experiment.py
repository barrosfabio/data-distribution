from distribution.train import train
from distribution.data import data_helpers
from distribution.hierarchy.lcpn_tree import LCPNTree
from distribution.hierarchy.lcn_tree import LCNTree
from distribution.hierarchy.hierarchical_constants import LCPN_CLASSIFIER, LCN_CLASSIFIER
from distribution.resampling.resampling_constants import FLAT_RESAMPLING
from distribution.resampling.resampling_algorithm import ResamplingAlgorithm
from distribution.data.singleton_frame import SingletonFrame

class ClassificationExperiment():

    def __init__(self, unique_classes, input_data, output_data, classifier_type, strategy, resampler):
        self.unique_classes = unique_classes
        self.input_data = input_data
        self.output_data = output_data
        self.classifier_type = classifier_type
        self.strategy = strategy
        self.resampler = resampler



    def classification(self, train_index, test_index):

        # Instantiate a tree
        if self.classifier_type == LCN_CLASSIFIER:
            tree = LCNTree(self.unique_classes, self.strategy, self.resampler)
        else:
            tree = LCPNTree(self.unique_classes, self.strategy, self.resampler)

        # Slice inputs and outputs
        inputs_train, outputs_train = self.input_data[train_index], self.output_data[train_index]
        inputs_test, outputs_test = self.input_data[test_index], self.output_data[test_index]

        if self.strategy == FLAT_RESAMPLING:
            resampling_algorithm = ResamplingAlgorithm(self.resampler, self.strategy, 1, 3)
            inputs_train, outputs_train = resampling_algorithm.resample(inputs_train, outputs_train)

        train_data_frame = data_helpers.array_to_data_frame(inputs_train, outputs_train)

        # Here we are creating one node for each sub-class
        tree.build_tree()

        singleton_df = SingletonFrame.instance()
        singleton_df.set_train_data(train_data_frame)

        # Retrieving data according to the policies
        tree.retrieve_data(tree.root)

        resampling_strategy = self.strategy + self.resampler

        # Check what is the data distribution here
        train.calculate_distribution(tree, resampling_strategy)
