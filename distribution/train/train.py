from distribution.config.global_config import GlobalConfig
from distribution.result.results_helpers import transform_multiple_dict_to_csv

def calculate_distribution(tree):
    #Retrieving Global experiment configurations
    global_config = GlobalConfig.instance()

    # Create a list of dictionaries to save the results for each resampler
    count_results_list = []

    # Retrieve list of resamplers
    # Traverse the tree to train the nodes
    tree.count_hierarchical(tree.root, count_results_list)

    # Save the resamplers_results_list
    transform_multiple_dict_to_csv(global_config.directory_list['resampling'], count_results_list, 'resamplers_results')