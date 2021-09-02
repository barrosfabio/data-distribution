import numpy as np
from distribution.hierarchy.class_helpers import find_parent
from distribution.policy.policy import Policy
from distribution.hierarchy.hierarchical_constants import CLASS_SEPARATOR


def find_sibling_classes(combinations, data_class, parent):
    # Classes at the same level
    same_level_classes = {value for value in combinations if
                          len(value.split(CLASS_SEPARATOR)) == len(data_class.split(CLASS_SEPARATOR))}

    # Classes at the same level and that share the same parent
    siblings = {value for value in same_level_classes if
                find_parent(value) == parent}

    return list(siblings)


def find_sibling_child(combinations, sibling):
    # Tries to find child classes for each sibling. Tests if the value is equal to sibling class + /
    child_classes = {value for value in combinations if value.find(sibling + CLASS_SEPARATOR) != -1}

    return list(child_classes)


class SiblingsPolicy(Policy):

    def find_classes_siblings_policy(self, combinations):
        # Find a list of the classes at the same level that share the same parent
        siblings = find_sibling_classes(combinations, self.current_class, self.parent_class)

        # Lists to store the positive and the negative classes
        positive_classes = []
        negative_classes = []

        # Find the child nodes for each of the sibling classes
        for sibling in siblings:

            child_classes = find_sibling_child(combinations, sibling)
            print('Class {} - Child Classes {}'.format(sibling, child_classes))

            # Save the child from the current class as positive
            if sibling == self.current_class:
                positive_classes = np.append(positive_classes, child_classes)

            # Save the child from the other classes as negative
            else:
                # Fix this
                # Negative classes = parent/*
                negative_classes = np.append(negative_classes, str(sibling))
                negative_classes = np.append(negative_classes, child_classes)

        # Appending the current class as positive as well
        positive_classes = np.append(positive_classes, self.current_class)

        self.positive_classes = positive_classes
        self.negative_classes = negative_classes
