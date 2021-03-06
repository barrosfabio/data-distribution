from distribution.hierarchy.generic_tree import Tree
from distribution.hierarchy.data import Data
from distribution.hierarchy.hierarchical_constants import NEGATIVE_CLASS
from distribution.resampling.resampling_constants import LOCAL_RESAMPLING, IR_SELECTIVE_RESAMPLING
from distribution.data.data_helpers import count_by_class_result
from distribution.resampling.resampling_algorithm import ResamplingAlgorithm
from distribution.data.data_helpers import save_data_frame

import numpy as np
import pandas as pd


def relabel_outputs_lcn(data_frame, data_class):
    unique_classes = np.unique(data_frame['class'])
    data_frame['class'] = data_frame['class'].replace(unique_classes, data_class)

    return data_frame


def count_negative_predictions(predictions):
    negative_class_string = NEGATIVE_CLASS
    count_negative = 0

    for i in range(0, len(predictions)):
        if predictions[i] == negative_class_string:
            count_negative += 1
    return count_negative


def find_predicted_class(predictions):
    negative_class_string = NEGATIVE_CLASS
    predicted_class = ''

    for i in range(0, len(predictions)):
        if predictions[i] != negative_class_string:
            predicted_class = predictions[i]
    return predicted_class


class LCNTree(Tree):

    def retrieve_data(self, root_node, train_data_frame):
        #print('Currently retrieving data for class: {}'.format(root_node.class_name))

        # If the current node doesn't have child, it is a leaf node
        if len(root_node.child) == 0:
            #print('Reached leaf node level, call is being returned.')
            return
        else:
            # Retrieve the number of children for the current node
            children = len(root_node.child)
            #print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Iterate over the current node child to call recursively for all of them
            for i in range(children):
                # Retrieving the current node
                #print('Current Child is {}'.format(root_node.child[i].class_name))
                current_node = root_node.child[i]

                # Add classes to

                # Retrieve the positive classes for the current node
                data_class_relationship = current_node.data_class_relationship
                positive_classes = data_class_relationship.positive_classes
                negative_classes = data_class_relationship.negative_classes
                #print('Positive classes {} for node {}'.format(positive_classes, current_node.class_name))
                #print('Negative classes {} for node {}'.format(negative_classes, current_node.class_name))

                # Retrieve the filtered data from the data_frame
                positive_classes_data = train_data_frame[train_data_frame['class'].isin(positive_classes)]
                negative_classes_data = train_data_frame[train_data_frame['class'].isin(negative_classes)]

                # Relabel the positive classes
                positive_classes_data = relabel_outputs_lcn(positive_classes_data, current_node.class_name)

                # Relabel the negative classes
                negative_classes_data = relabel_outputs_lcn(negative_classes_data, NEGATIVE_CLASS)

                # Concatenate both positive and negative classes data-frames
                frames = [positive_classes_data, negative_classes_data]
                final_array = pd.concat(frames)
                input_train = final_array.iloc[:,0:-1]
                output_train = final_array.iloc[:,-1]
                output_train = output_train.to_numpy()

                if self.strategy == LOCAL_RESAMPLING or self.strategy == IR_SELECTIVE_RESAMPLING:
                    unique_classes = np.unique(output_train)
                    if len(unique_classes) > 1:
                        resampling = ResamplingAlgorithm(self.resampling_algorithm, self.strategy, 1, 3)
                        [input_train, output_train] = resampling.local_resample_lcn(input_train, output_train, current_node.class_name)

                # Store the data in the node
                current_node.data = Data(input_train, output_train)

                # Save the data in a csv file
                save_data_frame(current_node.data, current_node.class_name)

                # Continue the process recursively for all
                self.retrieve_data(current_node, train_data_frame)

    def count_hierarchical(self, root_node):
        print('Training a LCN Classifier')

        # Retrieve child nodes
        children = len(root_node.child)

        if children == 0:
            print('This is a leaf node, we reached the end of the tree')
            return
        else:
            # Train each child node
            for i in range(children):
                # Retrieve node being visited
                visited_node = root_node.child[i]
                print('Child is {}'.format(visited_node.class_name))

                count_by_class_result(visited_node.class_name, self.strategy + '-' + self.resampling_algorithm, visited_node.data.outputs)

                print('Finished Training')

                # Go down the tree
                self.count_hierarchical(visited_node)
