from distribution.policy.policy import Policy
from distribution.hierarchy.hierarchical_constants import CLASS_SEPARATOR


def find_child_classes(combinations, current_class):
    # Tries to find child classes for each sibling. Tests if the value is equal to sibling class + /
    child_classes = {value for value in combinations if value.find(current_class + CLASS_SEPARATOR) != -1}

    return list(child_classes)

class LessInclusive(Policy):

    def __init__(self, current_class):
        self.current_class = current_class
        self.positive_classes = []
        self.negative_classes = []
        self.direct_child_classes = []

    def find_less_inclusive_classes(self, combinations):
        # Arrays to store positive and negative classes
        positive_classes = []
        negative_classes = []

        # Adding current class as positive
        positive_classes.append(self.current_class)

        # Appending the current class as positive
        positive_classes += find_child_classes(combinations, self.current_class)

        # Find the complement of the list
        for data_class in combinations:
            # All of the classes from all combinations that are not in the positive_classes will be negative, according to the policy
            if data_class not in positive_classes:
                negative_classes.append(data_class)



        # Saving positive and negative classes
        self.positive_classes = positive_classes
        self.negative_classes = negative_classes



