from distribution.hierarchy.hierarchical_constants import CLASS_SEPARATOR


def find_parent(possible_class):
    if len(possible_class) > 1:

        possible_class_splitted = possible_class.split(CLASS_SEPARATOR)
        possible_class_splitted = possible_class_splitted[0:-1]
        parent = CLASS_SEPARATOR.join(possible_class_splitted)

    # Parent is the root of the tree
    else:
        parent = 'R'
    return parent


def compare_child_length(possible_class, comparison):
    possible_class_splitted = possible_class.split(CLASS_SEPARATOR)
    comparison_list = comparison.split(CLASS_SEPARATOR)

    if len(comparison_list) == (len(possible_class_splitted) + 1):
        return True
    else:
        return False


