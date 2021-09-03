from abc import abstractmethod

from distribution.hierarchy.hierarchical_constants import POLICY
from distribution.policy.less_inclusive import LessInclusive
from distribution.hierarchy.node import Node
from distribution.policy.siblings_policy import SiblingsPolicy
from distribution.hierarchy.tree_helpers import get_possible_classes
from distribution.hierarchy.class_helpers import find_parent
from distribution.config.global_config import GlobalConfig
from distribution.hierarchy.hierarchical_constants import LCPN_CLASSIFIER, LCN_CLASSIFIER


def find_node(root, node_class):
    if root.class_name == node_class:
        return root
    # That means we are passing leaf nodes as the root
    elif root == None:
        return None
    else:
        children = len(root.child)

        for i in range(children):
            node = find_node(root.child[i], node_class)

            if node is not None:
                return node

class Tree():

    def __init__(self, unique_classes, strategy, resampling_algorithm):
        self.possible_classes = unique_classes
        self.strategy = strategy
        self.resampling_algorithm = resampling_algorithm
        self.root = None

    @abstractmethod
    def retrieve_data(self, root_node, data_frame):
        pass

    @abstractmethod
    def count_hierarchical(self, root_node, resamplers_results_list):
        pass


    def insert_node(self, root, parent, data_class_relationship, node_class):

        if root is None:
            root = Node(node_class, data_class_relationship)
        else:
            # If current root is the parent, then we add to its child
            if root.class_name == parent:
                root.child.append(Node(node_class, data_class_relationship))
            else:
                # If current root is not the parent, then we need to recursively go through the tree until we find its parent
                children = len(root.child)

                for i in range(children):
                    child_updated = self.insert_node(root.child[i], parent, data_class_relationship, node_class)
                    root.child[i] = child_updated

        return root

    def build_tree(self):
        root = None
        combinations = get_possible_classes(self.possible_classes)
        # Global Config
        global_config = GlobalConfig.instance()

        for i in range(len(combinations)):
            # Insert current node
            current_node = combinations[i]

            # Identify parent
            parent = find_parent(current_node)

            if global_config.local_classifier == LCPN_CLASSIFIER:
                policy = 'siblings'
            elif global_config.local_classifier == LCN_CLASSIFIER:
                policy = 'less-inclusive'


            if policy == 'siblings':

                # Builds an object to store the positive, negative classes and the direct_child of a given class
                data_classes_relationship = SiblingsPolicy(current_node, parent)

                # Identify Immediate child of the current class
                data_classes_relationship.find_direct_child(combinations)

                # Identify Positive and Negative Classes
                data_classes_relationship.find_classes_siblings_policy(combinations)

            elif policy == 'less-inclusive':

                # Instantiating the Less Inclusive Policy
                data_classes_relationship = LessInclusive(current_node)

                # Finding the positive and negative classes according to the Less Inclusive Policy
                data_classes_relationship.find_less_inclusive_classes(combinations)


            # Insert the node in the tree
            root = self.insert_node(root, parent, data_classes_relationship, current_node)

        self.root = root

