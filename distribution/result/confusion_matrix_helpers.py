import matplotlib
matplotlib.use('Agg')
import numpy as np
import itertools
from matplotlib import pyplot as plt
from distribution.config.global_config import GlobalConfig


"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(cm, classes, image_name,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(28, 28)):
    plt.rcParams.update({'font.size': 32})
    plt.figure(figsize=figsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm = cm.astype('int32')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(image_name, bbox_inches="tight")
    # plt.show(block=False)
    plt.close()


def calculate_all_folds_conf_matrix(conf_matrix_list, size):
    final_conf_matrix = np.empty([size, size])

    for cm in conf_matrix_list:
        final_conf_matrix += cm

    return final_conf_matrix


def calculate_average_confusion_matrix(conf_matrix_list, unique_classes):
    global_config = GlobalConfig.instance()

    # Calculating the metrics
    final_cm = calculate_all_folds_conf_matrix(conf_matrix_list, len(unique_classes))

    # Plotting final_cm for the experiment
    image_path = global_config.directory_list['overall_results']
    plot_confusion_matrix(final_cm, classes=unique_classes, image_name=f'{image_path}/conf_matrix.pdf',
                          normalize=False,
                          title='Confusion Matrix')
    plot_confusion_matrix(final_cm, classes=unique_classes, image_name=f'{image_path}/conf_matrix_normalized.pdf',
                          normalize=True,
                          title='Confusion Matrix')