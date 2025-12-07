from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, y, y_hat):
        raise NotImplementedError("Subclass of Metric should implement update")
    
    @abstractmethod
    def mean(self):
        raise NotImplementedError("Subclass of Metric should implement mean")
    
    @abstractmethod
    def stdev(self):
        raise NotImplementedError("Subclass of Metric should implement stdev")
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError("Subclass of Metric should implement reset")
    
    @abstractmethod
    def __call__(self):
        raise NotImplementedError("Subclass of Metric should implement __call__")