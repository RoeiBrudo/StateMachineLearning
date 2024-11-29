from abc import ABC, abstractmethod


class NoHiddenModelAbstract(ABC):
    """
    Model parameters:
    m: input dimension at each time
    """
    def __init__(self):
        self.is_hidden = False

    @abstractmethod
    def build_model(self, **kwargs):
        ...

    @abstractmethod
    def fit(self, **kwargs):
        ...

    @abstractmethod
    def predict_x_to_y(self, X, return_seq):
        """
        :param X: input matrix shape (None, m)
        :param return_seq: boolean
        :param kwargs:
        :return: if true, return sequence of y, shape (None,)
                else, return (1,)
        """
        ...

    @abstractmethod
    def save_model(self, path):
        ...

    @abstractmethod
    def load_model(self, path):
        ...

