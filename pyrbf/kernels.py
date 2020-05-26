from abc import ABCMeta
from abc import abstractmethod

import numpy as np

class Kernel():
    """Abstract kernel class."""
    __metaclass__ = ABCMeta
    def __init__(self, param = 1):
        """Creates an instance of Kernel.

        Parameters
        ----------
        param : float
            The :math:`c` parameter of the kernel. Default is 1.

        Attributes
        ----------
        
        """
        self.set_param(param)
        
    @property
    def param(self):
        """Getter for the kernel parameter :math:`c`.

        Returns
        -------
        param : float
            The :math:`c` parameter of the kernel.
        """
        return self._param
        
    @property
    def dmin(self):
        """Getter for the :math:`d_{\\rm min}` associated with the kernel.

        Returns
        -------
        dmin : int
            The :math:`d_{\\rm min}` associated with the kernel.
        """
        return self._dmin
        
    def set_param(self, param):
        """Setter for the kernel parameter :math:`c`.

        Parameters
        ---------
        param : float
            The :math:`c` parameter of the kernel.

        Returns
        -------
        None
        """
        self._param = param
        
    @classmethod
    def from_dict(cls, input_dict):
        """Build a Kernel instance from a dict.

        Parameters
        ----------
        input_dict : dict
            A dictionary decribing the kernel.

        Return
        ------
        kernel : Kernel
            An instance of Kernel or one of its subclasses.
        """
        class_name = input_dict["class"]
        param = input_dict["param"]
        return globals()[class_name](param)
        
    def to_dict(self):
        """Build a dict representation of the kernel.

        Parameters
        ----------

        Return
        ------
        output_dict : dict
            A dict representation of the kernel.
        """
        output_dict = {
            "class" : self.__class__.__name__,
            "param" : self.param
            }
        return output_dict
        
    def copy(self):
        """Build a copy of this kernel.

        Parameters
        ----------

        Return
        ------
        kernel : Kernel
            A copy of this kernel.
        """
        return Kernel.from_dict(self.to_dict())
        
    @abstractmethod
    def evaluate(self, dist):
        """Kernel evaluation.

        Parameters
        ----------
        dist : numpy.ndarray
            An array of distances.
            
        Return
        ------
        values : numpy.ndarray
            An array of the same shape as `dist`.
        """
        raise NotImplementedError()
        
    @abstractmethod
    def derivatives(self, dist):
        """Kernel derivative evaluation.

        Parameters
        ----------
        dist : numpy.ndarray
            An array of distances.
            
        Return
        ------
        values : numpy.ndarray
            An array of the same shape as `dist`.
        """
        raise NotImplementedError()
        
class LinearKernel(Kernel):
    """
    The value of the kernel for a distance :math:`d` is computed as
    
    .. math::
        
        \phi_c(d) = cd
        
    and its derivative is computed as
        
    .. math::
        
        \phi_c^\prime(d) = c
    """
    def __init__(self, param = 1):
        Kernel.__init__(self, param)
        self._dmin = 0
        
    def evaluate(self, dist):
        return self.param * dist
        
    def derivatives(self, dist):
        return self.param * np.ones_like(dist)
        
class CubicKernel(Kernel):
    """
    The value of the kernel for a distance :math:`d` is computed as
    
    .. math::
        
        \phi_c(d) = cd^3
        
    and its derivative is computed as
        
    .. math::
        
        \phi_c^\prime(d) = 2cd^2
    """
    def __init__(self, param = 1):
        Kernel.__init__(self, param)
        self._dmin = 1
        
    def evaluate(self, dist):
        return self.param * dist**3
        
    def derivatives(self, dist):
        return 2 * self.param * dist**2
        
class ThinPlateKernel(Kernel):
    """
    The value of the kernel for a distance :math:`d` is computed as
    
    .. math::
        
        \phi_c(d) = d^2 {\\rm log}(cd)
        
    and its derivative is computed as
        
    .. math::
        
        \phi_c^\prime(d) = d (1 + 2{\\rm log}(cd))
    """
    def __init__(self, param = 1):
        Kernel.__init__(self, param)
        self._dmin = 1
        
    def evaluate(self, dist):
        values = np.zeros_like(dist)
        non_zero = dist > 0
        if np.any(non_zero):
            values[non_zero] = dist[non_zero]**2 * np.log(self.param * dist[non_zero])
        return values
        
    def derivatives(self, dist):
        values = np.zeros_like(dist)
        non_zero = dist > 0
        if np.any(non_zero):
            values[non_zero] = dist[non_zero] * (1 + 2*np.log(self.param * dist[non_zero]))
        return values
        
class GaussianKernel(Kernel):
    """
    The value of the kernel for a distance :math:`d` is computed as
    
    .. math::
        
        \phi_c(d) = e^{- (\\frac{d}{c})^2}
        
    and its derivative is computed as
        
    .. math::
        
        \phi_c^\prime(d) = -2 \\frac{d}{c^2} e^{-(\\frac{d}{c})^2}
    """
    def __init__(self, param = 1):
        Kernel.__init__(self, param)
        self._dmin = -1
        
    def evaluate(self, dist):
        return np.exp(- (dist/self.param)**2)
        
    def derivatives(self, dist):
        return -2 * dist/self.param**2 *  np.exp(-(dist/self.param)**2)
        
class MultiQuadraticKernel(Kernel):
    """
    The value of the kernel for a distance :math:`d` is computed as
    
    .. math::
        
        \phi_c(d) = \sqrt{1 + (cd)^2}
        
    and its derivative is computed as
        
    .. math::
        
        \phi_c^\prime(d) = \\frac{d c^2}{\sqrt{1 + (cd)^2}}
    """
    def __init__(self, param = 1):
        Kernel.__init__(self, param)
        self._dmin = 0
        
    def evaluate(self, dist):
        return np.sqrt(1 + (self.param * dist)**2)
        
    def derivatives(self, dist):
        return self.param**2 * dist / np.sqrt(1 + (self.param * dist)**2)
        
class InverseQuadraticKernel(Kernel):
    """
    The value of the kernel for a distance :math:`d` is computed as
    
    .. math::
        
        \phi_c(d) = \\frac{1}{1 + (cd)^2}
        
    and its derivative is computed as
        
    .. math::
        
        \phi_c^\prime(d) = \\frac{-2 d c^2}{(1 + (cd)^2)^2}
    """
    def __init__(self, param = 1):
        Kernel.__init__(self, param)
        self._dmin = -1
        
    def evaluate(self, dist):
        return 1/(1 + (self.param * dist)**2)
        
    def derivatives(self, dist):
        return -2 * self.param**2 * dist / (1 + (self.param * dist)**2)**2
        
class InverseMultiQuadraticKernel(Kernel):
    """
    The value of the kernel for a distance :math:`d` is computed as
    
    .. math::
        
        \phi_c(d) = \\frac{1}{\sqrt{1 + (cd)^2}}
        
    and its derivative is computed as
        
    .. math::
        
        \phi_c^\prime(d) = \\frac{- d c^2}{\sqrt{1 + (cd)^2}^3}
    """
    def __init__(self, param = 1):
        Kernel.__init__(self, param)
        self._dmin = -1
        
    def evaluate(self, dist):
        return 1/np.sqrt(1 + (self.param * dist)**2)
        
    def derivatives(self, dist):
        return - self.param**2 * dist / np.sqrt(1 + (self.param * dist)**2)**3
    
kernels_dict = {
    "linear"                    : LinearKernel,
    "cubic"                     : CubicKernel,
    "thin-plate"                : ThinPlateKernel,
    "gaussian"                  : GaussianKernel,
    "multi-quadratic"           : MultiQuadraticKernel,
    "inverse-quadratic"         : InverseQuadraticKernel,
    "inverse-multi-quadratic"   : InverseMultiQuadraticKernel
}