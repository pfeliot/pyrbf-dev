import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import linalg
from scipy.optimize import minimize

from .kernels import Kernel, kernels_dict
from .tails import Tail, tails_dict

class RBF():
    def __init__(self, kernel = None, tail = None, eta = 1e-12):
        """Creates an instance of RBF model.

        Parameters
        ----------
        kernel : Kernel or str, optional
            The model's kernel. If `None` is passed, a CubicKernel with `param = 1` is used.
            Possible str values are "linear", "cubic", "thin-plate", "gaussian", 
            "multi-quadratic", "inverse-quadratic" and "inverse-multi-quadratic".
            
        tail : Tail or str, optional
            The model's tail. If `None` is passed, a tail with degree given by the kernel's :math:`d_\mathrm{min}` is used.
            Possible str values are "no-tail", "constant" and "linear".
            
        eta : float
            A small positive value that is added to the diagonal of the model's matrix to improve its conditioning. 
            Default is `1e-6`.
        """
        self._set_kernel(kernel)
        self._set_tail(tail)
        self._set_eta(eta)
        
    def _set_kernel(self, kernel):
        """Setter for the model's kernel"""
        if kernel is None:
            self._kernel = kernels_dict["cubic"](1)
        elif isinstance(kernel, str):
            self._kernel = kernels_dict[kernel.lower()](1)
        else:
            self._kernel = kernel
        
    def _set_tail(self, tail):
        """Setter for the model's tail."""
        if tail is None:
            self._set_tail_default()
        elif isinstance(tail, str):
            self._set_tail_str(tail)
        else:
            if tail.degree >= self._kernel.dmin:
                self._tail = tail
            else:
                raise ValueError("Tail's degree ({}) should be larger than kernel's dmin ({}).".format(tail.degree, self._kernel.dmin))
                
    def _set_tail_default(self):
        """Default setter for the model's tail."""
        if self._kernel.dmin == -1:
            self._tail = tails_dict["no-tail"]()
        elif self._kernel.dmin == 0:
            self._tail = tails_dict["constant"]()
        elif self._kernel.dmin == 1:
            self._tail = tails_dict["linear"]()
        else:
            raise NotImplementedError()
            
    def _set_tail_str(self, tail):
        """String setter for the model's tail."""
        if tail.lower() in tails_dict:
            _tail = tails_dict[tail.lower()]()
            if _tail.degree >= self._kernel.dmin:
                self._tail = _tail
            else:
                raise ValueError("Tail's degree ({}) should be larger than kernel's dmin ({}).".format(_tail.degree, self._kernel.dmin))
        else:
            raise ValueError("Unknown tail '{}'.".format(tail))
        
    def _set_eta(self, eta):
        """Setter for the eta parameter."""
        self._eta = float(eta)
        
    def _is_fitted(self):
        """Return True if the model has been fitted and False otherwise."""
        return hasattr(self, "_lambda")
        
    @property
    def loo_residuals(self):
        """Leave-one-out residuals.

        Returns
        -------
        residuals : numpy.array or None
            The leave-one-out residuals associated with the model. `None` is returned if the model has not been fitted.
        """
        if self._is_fitted():
            return self._loo_residuals
        else:
            return None
        
    @classmethod
    def from_dict(cls, input_dict):
        """Create a model from a dict.
        
        Parameters
        ----------
        input_dict : dict
            A dictionnary describing a `RBF` model.
            
        Returns
        -------
        model : RBF
            A `RBF` model.
        """
        kernel = Kernel.from_dict(input_dict["kernel"])
        tail = Tail.from_dict(input_dict["tail"])
        eta = input_dict["eta"]
        ins = cls(kernel, tail, eta)
        if "lambda" in input_dict:
            ins._X = np.array(input_dict["X"])
            ins._y = np.array(input_dict["y"])
            ins._lambda = np.array(input_dict["lambda"])
            ins._LU = np.array(input_dict["LU"], ndmin = 2)
            ins._piv = np.array(input_dict["piv"])
            ins._loo_residuals = np.array(input_dict["loo"])
        return ins
        
    def to_dict(self):
        """Serialization to dict.
            
        Returns
        -------
        output_dict : dict
            A dict representation of the model.
        """
        output_dict = {
            "kernel" : self._kernel.to_dict(),
            "tail" : self._tail.to_dict(),
            "eta" : self._eta
            }
        if self._is_fitted():
            output_dict["X"] = self._X.tolist()
            output_dict["y"] = self._y.tolist()
            output_dict["lambda"] = self._lambda.tolist()
            output_dict["LU"] = self._LU.tolist()
            output_dict["piv"] = self._piv.tolist()
            output_dict["loo"] = self._loo_residuals.tolist()
        return output_dict
        
    def copy(self):
        """Model copy.
            
        Returns
        -------
        model : RBF
            A copy of this model.
        """
        ins = self.__class__(self._kernel.copy(), self._tail.copy(), self._eta)
        if self._is_fitted():
            ins._X = np.array(self._X)
            ins._y = np.array(self._y)
            ins._lambda = np.array(self._lambda)
            ins._LU = np.array(self._LU)
            ins._piv = np.array(self._piv)
            ins._loo_residuals = np.array(self._loo_residuals)
        return ins
        
    def predict(self, X):
        """Predictions of the model at `X`.
        
        Parameters
        ----------
        X : numpy.ndarray
            A matrix of points where the predictions are to be computed.
            
        Returns
        -------
        y : numpy.array
            The model's predictions at `X`.
            
        s : numpy.array
            The model's predictions errors at `X`.
        """
        if not self._is_fitted():
            raise ValueError("Predictions with unfitted model.")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape((-1,1))
            
        D = cdist(X, self._X)
        Phix = self._kernel.evaluate(D)
        Px = self._tail.compute_P(X)
        y = np.dot(self._lambda, np.transpose(Phix)) + self._tail.evaluate(X)
        s = self._compute_std(Phix, Px)
        return y, s
        
    def sample_y(self, X, N): # TODO : Add some covariance structure here...
        if not self._is_fitted():
            raise ValueError("Predictions with unfitted model.")
            
        p, s = self.predict(X)
        return np.random.multivariate_normal(p, np.diag(s), N)
        
    def train(self, X, y, **kwargs):
        """Optimization of the model's kernel parameter.
        
        Parameters
        ----------
        X : numpy.ndarray
            The training points.
            
        y : numpy.array
            The associated output values.
            
        kwargs : optional
            Keyword arguments are passed to `scipy.optimize.minimize`.
            
        Returns
        -------
        loo : numpy.array
            The leave-one-out residuals after optimization.
        """
        x0 = np.array(1) if "x0" not in kwargs else np.array()
        bounds = [(0.01, 2)] if "bounds" not in kwargs else list(kwargs.pop("bounds"))
        res = minimize(lambda param : self.fit(X, y, param), x0, bounds = bounds, **kwargs)
        loo = self.fit(X, y, res.x[0])
        return loo
    
    def fit(self, X, y, param = None):
        """Fitting of the model.
        
        Parameters
        ----------
        X : numpy.ndarray
            The training points.
            
        y : numpy.array
            The associated output values.
            
        param : float, optional
            A value for the kernel's parameter. If `None` is passed, the kernel is unchanged.
            
        Returns
        -------
        loo : float
            The l2-norm of the leave-one-out residuals vector.
        """
        # Check X and y
        self._X, self._y = self._check_X_y(X, y)
        
        # Update kernel parameter if required
        if param is not None:
            self._kernel.set_param(param)
            
        # Assemble and solve interpolation problem
        N = self._X.shape[0]
        K = self._K(self._X)
        self._LU, self._piv = linalg.lu_factor(K)
        b = self._tail.compute_b(self._y, d = self._X.shape[1])
        x = linalg.lu_solve((self._LU, self._piv), b)
        
        # Set internal parameters
        x = x.ravel()
        self._lambda = x[:N]
        self._tail.set_params(x[N:])
        
        # Compute LOO residuals
        Kinv = linalg.lu_solve((self._LU, self._piv), np.eye(K.shape[0]))
        self._loo_residuals = self._lambda/np.diag(Kinv)[:N]
        return linalg.norm(self._loo_residuals)
        
    #---------------------------------------------------------------------------------
    def _check_X_y(self, X, y):
        arr = np.array(X)
        X = arr.reshape((-1,1)) if arr.ndim == 1 else arr
        y = np.array(y).ravel()
        if X.shape[0] != len(y):
            raise ValueError("Missmatching shapes between X {} and y {}.".format(X.shape, y.shape))
        return X, y
        
    def _K(self, X):
        P = self._tail.compute_P(X)
        Phi = self._Phi(X)
        if P is not None:
            A0 = np.hstack((Phi + self._eta * np.eye(Phi.shape[0]), P))
            A1 = np.hstack((P.T, self._eta * np.eye(P.shape[1])))
            return np.vstack((A0, A1))
        else:
            return Phi + self._eta * np.eye(Phi.shape[0])
        
    def _Phi(self, X):
        dist = pdist(X)
        phi = self._kernel.evaluate(dist)
        return squareform(phi, force = "tomatrix") + self._kernel.evaluate(0) * np.eye(X.shape[0])
            
    def _compute_std(self, ux, Px):
        M = ux.shape[0]
        u = ux if Px is None else np.hstack((ux, Px))
        _temp1 = linalg.lu_solve((self._LU, self._piv), np.transpose(u))
        _temp2 = u.ravel("C") * _temp1.ravel("F")
        _temp3 = np.sum(_temp2.reshape((M,-1)), axis = 1)
        return np.sqrt(np.abs(self._kernel.evaluate(np.zeros(1)) * np.ones(M) - _temp3))
        
