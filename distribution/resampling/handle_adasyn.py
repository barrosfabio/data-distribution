import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing


def calculate_adasyn_possible(input_data, output_data, sampling_strategy_items, n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)

    samples_to_generate = {}

    for class_sample, n_samples in sampling_strategy_items.items():
        if n_samples == 0:
            continue
        target_class_indices = np.flatnonzero(output_data == class_sample)
        X_class = _safe_indexing(input_data, target_class_indices)

        knn.fit(input_data)

        # calculating the nearest neighbors of a class
        nns = knn.kneighbors(X_class, return_distance=False)[:, 1:]

        # The ratio is computed using a one-vs-rest manner. Using majority
        # in multi-class would lead to slightly different results at the
        # cost of introducing a new parameter.
        n_neighbors = knn.n_neighbors - 1
        ratio_nn = np.sum(output_data[nns] != class_sample, axis=1) / n_neighbors

        if not np.sum(ratio_nn):
            continue

        ratio_nn /= np.sum(ratio_nn)
        n_samples_generate = np.rint(ratio_nn * n_samples).astype(int)

        # rounding may cause new amount for n_samples
        n_samples = np.sum(n_samples_generate)

        if n_samples != 0:
            samples_to_generate[class_sample] = n_samples

    return samples_to_generate

def count_class_sample(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))

def count_not_majority(y):
    target_stats = count_class_sample(y)
    n_sample_majority = max(target_stats.values())
    class_majority = max(target_stats, key=target_stats.get)

    sampling_strategy = {
        key: n_sample_majority - value
        for (key, value) in target_stats.items()
        if key != class_majority
    }

    return sampling_strategy

def total_samples(samples_to_generate, max_count):
    for key, value in samples_to_generate.items():
        samples_to_generate[key] = max_count

    return samples_to_generate


def handle_adasyn(input_data, output_data, n_neighbors):

    # Checking how many samples we need to generate for each class
    classes_to_resample = count_not_majority(output_data)

    # Checking which classes need to be resampled based on ADASYN rules
    samples_to_generate = calculate_adasyn_possible(input_data, output_data, classes_to_resample, n_neighbors)

    # Count the dataset instances for each class in the dataset
    count_stats = count_class_sample(output_data)

    # Retrieving how many samples will be generated for each of the resampled classes. For default we are using max count
    samples_to_generate = total_samples(samples_to_generate, max(count_stats.values()))

    # returning how many samples we want to generate for each class in a dictionary
    return samples_to_generate
