from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from distribution.result.proba_result import ProbaResult
from distribution.classification.classification_constants import *


import numpy as np


class ClassificationAlgorithm:

    def __init__(self, option):
        self.classifier_name = option
        if option == RANDOM_FOREST:
            self.classifier = RandomForestClassifier(n_estimators=150, criterion='gini', n_jobs=-1)
        elif option == MLP:
            self.classifier = MLPClassifier(solver='lbfgs',activation='relu', max_iter=1000)
            #self.classifier = MLPClassifier(activation='relu', hidden_layer_sizes=(100,100), max_iter=1000)
        elif option == SVM:
            self.classifier = SVC(kernel='rbf', C=1.0, probability=True)
        elif option == DECISION_TREE:
            self.classifier = DecisionTreeClassifier(criterion = 'gini')
        elif option == NAIVE_BAYES:
            self.classifier = GaussianNB()
        elif option == KNN:
            self.classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        elif option == ADA_BOOST:
            base_estimator = DecisionTreeClassifier(criterion='gini')
            self.classifier = AdaBoostClassifier(base_estimator = base_estimator, n_estimators=150)
        elif option == BAGGING:
            base_estimator = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            self.classifier = BaggingClassifier(base_estimator = base_estimator, max_samples=0.5, max_features=0.5)
        elif option == EXTRA_TREES:
            self.classifier = ExtraTreesClassifier(n_estimators=150, criterion='gini', n_jobs=-1)
        elif option == GRADIENT_BOOST:
            self.classifier = GradientBoostingClassifier(n_estimators=150, learning_rate=1.0, max_depth=1)
        elif option == VOTING:
            clf1 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            clf2 = ExtraTreesClassifier(n_estimators=150, criterion='gini', n_jobs=-1)
            voting = VotingClassifier(estimators = [('svm', clf1), ('ext',clf2)],voting = 'soft', n_jobs=4)
            self.classifier = voting
        elif option == STACKING:
            clf1 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            clf2 = ExtraTreesClassifier(n_estimators=150, criterion='gini', n_jobs=-1)
            estimators = [('svm', clf1), ('ext', clf2)]
            self.classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=4)


    def train(self, input_train, output_train):
        # Training the classifier
        self.classifier = self.classifier.fit(input_train, output_train)

    def prediction_single_record(self, data):
        data = data.reshape(1, -1)
        predicted_class = self.classifier.predict(data)

        return predicted_class

    def prediction(self, input_test):
        predicted_classes = self.classifier.predict(input_test)

        return predicted_classes


    def prediction_proba(self, data):
        data = data.reshape(1, -1)
        predicted_class = self.classifier.predict(data)
        predicted_proba = self.classifier.predict_proba(data)
        predicted_proba = predicted_proba[0]

        possible_classes = self.classifier.classes_

        # Find index of predicted class and save this index only
        index = np.where(possible_classes == predicted_class)
        index = index[0]

        proba_predicted_class = predicted_proba[index[0]]
        result = ProbaResult(predicted_class, proba_predicted_class)
        return result

