from abc import ABC, abstractmethod


class HiddenModelAbstract(ABC):
    """
    Model parameters:
    m: input dimension at each time
    """
    def __init(self):
        self.is_hidden = True

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
    def predict_x_to_hidden(self, X, return_seq=True):
        ...

    @abstractmethod
    def predict_hidden_to_y(self, X, H, return_seq=True):
        ...

    @abstractmethod
    def save_model(self, path):
        ...

    @abstractmethod
    def load_model(self, path):
        ...

