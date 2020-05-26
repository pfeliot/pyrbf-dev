from abc import ABCMeta
from abc import abstractmethod

import numpy as np

class Tail():
    """Abstract tail class."""
    __metaclass__ = ABCMeta
    def __init__(self):
        pass
        
    @property
    def params(self):
        return None
        
    def set_params(self, params):
        pass
        
    @classmethod
    def from_dict(cls, input_dict):
        class_name = input_dict["class"]
        params = input_dict["params"]
        ins = globals()[class_name]()
        ins.set_params(params)
        return ins
        
    def to_dict(self):
        output_dict = {
            "class" : self.__class__.__name__,
            "params" : self.params
            }
        return output_dict
        
    def copy(self):
        return Tail.from_dict(self.to_dict())
        
    @abstractmethod
    def evaluate(self, X):
        raise NotImplementedError()
        
    @abstractmethod
    def compute_b(self, y, d = 1):
        raise NotImplementedError()
        
    @abstractmethod
    def compute_P(self, X):
        raise NotImplementedError()
        
class NoTail(Tail):
    def __init__(self):
        Tail.__init__(self)
        
    @property
    def degree(self):
        return -1
        
    @property
    def params(self):
        return None
        
    def set_params(self, params):
        pass
        
    def evaluate(self, X):
        return np.zeros(X.shape[0])
        
    def compute_b(self, y, d = 1):
        return y.reshape((-1,1))
        
    def compute_P(self, X):
        return None
        
class ConstantTail(Tail):
    def __init__(self):
        Tail.__init__(self)
        self._p = 0
        
    @property
    def degree(self):
        return 0
        
    @property
    def params(self):
        return self._p
        
    def set_params(self, params):
        self._p = float(params)
        
    def evaluate(self, X):
        return self._p * np.ones(X.shape[0])
        
    def compute_b(self, y, d = 1):
        return np.hstack((y, np.zeros(1))).reshape((-1,1))
        
    def compute_P(self, X):
        return np.ones((X.shape[0], 1))
        
class LinearTail(Tail):
    def __init__(self):
        Tail.__init__(self)
        self._p = np.zeros(1)
        self._c = 0
        
    @property
    def degree(self):
        return 1
        
    @property
    def params(self):
        params = list(self._p)
        params.append(self._c)
        return params
        
    def set_params(self, params):
        self._p = np.array(params[:-1])
        self._c = float(params[-1])
        
    def evaluate(self, X):
        return np.dot(X, self._p.reshape((-1,1))).ravel() + self._c
        
    def compute_b(self, y, d = 1):
        return np.hstack((y, np.zeros(d + 1))).reshape((-1,1))
            
    def compute_P(self, X):
        return np.hstack((X, np.ones((X.shape[0],1))))
        
tails_dict = {
    "no-tail" : NoTail,
    "constant" : ConstantTail,
    "linear" : LinearTail
    }