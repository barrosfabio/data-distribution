from distribution.config.global_config import GlobalConfig

def calculate_distribution(tree, strategy):

    # Retrieve list of resamplers
    # Traverse the tree to train the nodes
    tree.count_hierarchical(tree.root)
