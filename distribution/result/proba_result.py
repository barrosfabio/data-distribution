class ProbaResult(object):

    def __init__(self, predicted_class, predicted_probability):
        self.predicted_class = predicted_class
        self.predicted_probability = predicted_probability