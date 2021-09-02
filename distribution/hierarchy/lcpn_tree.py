from distribution.hierarchy.generic_tree import Tree
from distribution.data.data_helpers import slice_data
from distribution.hierarchy.data import Data
from distribution.hierarchy.hierarchical_constants import CLASS_SEPARATOR, PREDICTION_CONFIG
from distribution.data.data_helpers import slice_and_split_data_holdout
from distribution.resampling.resampling_algorithm import ResamplingAlgorithm
from distribution.data.data_helpers import count_by_class
from distribution.resampling.resampling_constants import LOCAL_RESAMPLING
from distribution.data.data_helpers import slice_data, array_to_data_frame
import numpy as np
import pandas as pd

def relabel_outputs_lcpn(positive_classes_data, data_class):
    # Slice dataset in inputs and outputs
    [input_data, output_data] = slice_data(positive_classes_data)

    class_splitted = data_class.split(CLASS_SEPARATOR)
    relabeled_outputs = []
    final_data_frame = pd.DataFrame(input_data)

    for sample in output_data:
        sample_splitted = sample.split(CLASS_SEPARATOR)
        sample_splitted = sample_splitted[0:len(class_splitted) + 1]

        relabeled_sample = CLASS_SEPARATOR.join(sample_splitted)
        relabeled_outputs.append(relabeled_sample)

    final_data_frame['class'] = relabeled_outputs
    return final_data_frame


class LCPNTree(Tree):

    def retrieve_data(self, root_node, train_data_frame):

        print('Currently retrieving dataset for class: {}'.format(root_node.class_name))

        # If the current node doesn't have child, it is a leaf node
        if len(root_node.child) == 0:
            print('Reached leaf node level, call is being returned.')
            return
        else:
            # Retrieve the positive classes for the current node
            data_class_relationship = root_node.data_class_relationship
            positive_classes = data_class_relationship.positive_classes
            print('Positive classes {} for node {}'.format(positive_classes, root_node.class_name))

            # Retrieve the filtered dataset from the data_frame
            positive_classes_data = train_data_frame[train_data_frame['class'].isin(positive_classes)]

            # Relabel the outputs to the child classes
            relabeled_data = relabel_outputs_lcpn(positive_classes_data, root_node.class_name)

            [input_train, output_train] = slice_data(relabeled_data)

            if self.strategy == LOCAL_RESAMPLING:
                unique_classes = np.unique(positive_classes_data.iloc[:,-1])
                if len(unique_classes) > 1:
                    resampling = ResamplingAlgorithm(self.resampling_algorithm, 1, 3)
                    [input_train, output_train] = resampling.resample(input_train, output_train)

            # Store the dataset in the node
            root_node.data = Data(input_train, output_train)

            # Retrieve the number of children for the current node
            children = len(root_node.child)
            print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Iterate over the current node child to call recursively for all of them
            for i in range(children):
                print('Child is {}'.format(root_node.child[i].class_name))
                self.retrieve_data(root_node.child[i], train_data_frame)

    def count_hierarchical(self, root_node, resamplers, count_results_list):

        print('Training a LCPN Classifier')

        # Testing if the node is leaf
        if len(root_node.child) == 0:
            print('This is a leaf node we do not need to train')
            return
        else:

            # Retrieve the number of children for the current node
            children = len(root_node.child)
            print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Testing if there are at least two child classes, Otherwise we don't need to train the classifier
            if (children > 1):

                count = count_by_class(root_node.class_name, self.strategy +'-'+self.resampling_algorithm, root_node.data.outputs)

                # Storing the results obtained by each resampler
                count_results_list.append(count)

            else:
                print('Current Node doesnt have multiple child, we dont need to train the classifier.')

            # Iterate over the current node child to call recursively for all of them
            for i in range(children):
                print('Child is {}'.format(root_node.child[i].class_name))
                self.count_hierarchical(root_node.child[i], resamplers, count_results_list)