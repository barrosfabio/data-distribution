class GlobalConfig():

    _instance = None

    def __init__(self):
        self.directory_list = None
        self.k_neighbors = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_directory_configuration(self, directory_list):
        self.directory_list = directory_list

    def set_kfold(self, kfold):
        self.kfold = kfold

    def set_local_classifier(self, local_classifier):
        self.local_classifier = local_classifier

    def set_metric(self, metric):
        self.metric = metric

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def set_resamplers(self, resamplers):
        self.resamplers = resamplers

    def set_classifier(self, classifier):
        self.classifier = classifier

    def set_resampler_results(self, resampler_results):
        self.resampler_results = resampler_results