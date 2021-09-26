class SingletonFrame():

    _instance = None

    def __init__(self):
        self.train_data_frame = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_train_data(self, train_data_frame):
        self.train_data_frame = train_data_frame