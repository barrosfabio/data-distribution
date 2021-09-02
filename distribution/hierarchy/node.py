class Node:
    is_parent = False
    is_leaf = False

    def __init__(self, class_name, data_class_relationship):
        self.class_name = class_name
        self.data_class_relationship = data_class_relationship
        self.child = []
        self.data = None
        self.classifier = None
        self.sub_train = None
        self.sub_validation = None

    def set_classifier(self, classifier):
        self.classifier = classifier

    def set_predicted_proba(self, predicted_proba):
        self.predicted_proba = predicted_proba
