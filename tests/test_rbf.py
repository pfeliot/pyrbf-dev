import numpy as np
import pytest

from pyrbf import RBF
from pyrbf.tails import tails_dict
from pyrbf.kernels import kernels_dict

def func_1d(x):
    return 5 + 3*np.sin(x)

def func_2d(x):
    return 3 + np.sin(x[:,0]) * np.sin(x[:,0])

class TestRBF:
    def test___init__defaults(self):
        model = RBF()
        assert isinstance(model._kernel, kernels_dict["cubic"])
        assert isinstance(model._tail, tails_dict["linear"])
        assert model._eta == 1e-6
        
    def test___init__kernels(self):
        for name, cls in kernels_dict.items():
            model = RBF(kernel = name)
            assert isinstance(model._kernel, cls)
            kernel = cls(2)
            model = RBF(kernel)
            assert isinstance(model._kernel, cls)
            assert model._kernel.param == 2
            assert model._tail.degree == kernel.dmin
        
    def test___init__tails(self):
        for name, cls in tails_dict.items():
            _tail = cls()
            for kernel in kernels_dict.values():
                k = kernel()
                if _tail.degree >= k.dmin:
                    model = RBF(k, tail = name)
                    assert isinstance(model._tail, cls)
                    model = RBF(k, tail = cls())
                    assert isinstance(model._tail, cls)
                else:
                    with pytest.raises(ValueError):
                        model = RBF(k, tail = name)
                    with pytest.raises(ValueError):
                        model = RBF(k, tail = cls())
        
    def test___init__eta(self):
        model = RBF(eta = 1)
        assert model._eta == 1
    
    def test_loo_residuals_1d(self):
        x = np.linspace(0, 2*np.pi, num = 8)
        y = func_1d(x)
        # model = RBF(x, y)
        model = RBF()
        assert model.loo_residuals is None
        model.fit(x, y)
        assert model.loo_residuals.shape == (8,)
    
    def test_loo_residuals_2d(self):
        x = 2*np.pi*np.random.random_sample((20, 2))
        y = func_2d(x)
        # model = RBF(x, y)
        model = RBF()
        assert model.loo_residuals is None
        model.fit(x, y)
        assert model.loo_residuals.shape == (20,)
    
    def test_from_dict_nofit(self):
        eta = 1
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                input_dict = {
                    "eta" : eta,
                    "kernel" : k.to_dict(),
                    "tail" : t.to_dict()
                    }
                if t.degree >= k.dmin:
                    model = RBF.from_dict(input_dict)
                    assert model._eta == eta
                    assert isinstance(model._kernel, kernel)
                    assert model._kernel.param == 2
                    assert isinstance(model._tail, tail)
                else:
                    with pytest.raises(ValueError):
                        model = RBF.from_dict(input_dict)
    
    def test_from_dict_fit_1d(self):
        x = np.linspace(0, 2*np.pi, num = 8)
        y = func_1d(x)
        eta = 1e-3
        p = 3
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    _model = RBF(k, t, eta)
                    _model.fit(x, y, p)
                    input_dict = {
                        "X" : x.reshape((-1,1)).tolist(),
                        "y" : y.tolist(),
                        "eta" : eta,
                        "kernel" : k.to_dict(),
                        "lambda" : _model._lambda.tolist(),
                        "LU" : _model._LU.tolist(),
                        "piv" : _model._piv.tolist(),
                        "loo" : _model._loo_residuals.tolist(),
                        "tail" : _model._tail.to_dict()
                        }
                    model = RBF.from_dict(input_dict)
                    assert model._X.shape == (8,1)
                    assert np.allclose(model._X.ravel(), x)
                    assert model._y.shape == (8,)
                    assert np.allclose(model._y, y)
                    assert model._eta == eta
                    assert isinstance(model._kernel, kernel)
                    assert model._kernel.param == p
                    assert np.allclose(model._lambda, _model._lambda)
                    assert np.allclose(model._LU, _model._LU)
                    assert np.allclose(model._piv, _model._piv)
                    assert np.allclose(model._loo_residuals, _model._loo_residuals)
                    assert isinstance(model._tail, _model._tail.__class__)
                    assert np.all(model._tail.params == _model._tail.params)
    
    def test_from_dict_fit_2d(self):
        x = 2*np.pi*np.random.random_sample((20, 2))
        y = func_2d(x)
        eta = 1e-3
        p = 3
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    _model = RBF(k, t, eta)
                    _model.fit(x, y, p)
                    input_dict = {
                        "X" : x.tolist(),
                        "y" : y.tolist(),
                        "eta" : eta,
                        "kernel" : k.to_dict(),
                        "lambda" : _model._lambda.tolist(),
                        "LU" : _model._LU.tolist(),
                        "piv" : _model._piv.tolist(),
                        "loo" : _model._loo_residuals.tolist(),
                        "tail" : _model._tail.to_dict()
                        }
                    model = RBF.from_dict(input_dict)
                    assert model._X.shape == (20,2)
                    assert np.allclose(model._X, x)
                    assert model._y.shape == (20,)
                    assert np.allclose(model._y, y)
                    assert model._eta == eta
                    assert isinstance(model._kernel, kernel)
                    assert model._kernel.param == p
                    assert np.allclose(model._lambda, _model._lambda)
                    assert np.allclose(model._LU, _model._LU)
                    assert np.allclose(model._piv, _model._piv)
                    assert np.allclose(model._loo_residuals, _model._loo_residuals)
                    assert isinstance(model._tail, _model._tail.__class__)
                    assert np.all(model._tail.params == _model._tail.params)
    
    def test_to_dict_nofit(self):
        eta = 1
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    output_dict = model.to_dict()
                    assert "eta" in output_dict
                    assert "kernel" in output_dict
                    assert "tail" in output_dict
                    assert output_dict["eta"] == model._eta
                    assert output_dict["kernel"] == model._kernel.to_dict()
                    assert output_dict["tail"] == model._tail.to_dict()
    
    def test_to_dict_fit_1d(self):
        x = np.linspace(0, 2*np.pi, num = 8)
        y = func_1d(x)
        eta = 1e-3
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y)
                    output_dict = model.to_dict()
                    assert "X" in output_dict
                    assert "y" in output_dict
                    assert "eta" in output_dict
                    assert "kernel" in output_dict
                    assert "tail" in output_dict
                    assert "lambda" in output_dict
                    assert "LU" in output_dict
                    assert "piv" in output_dict
                    assert "loo" in output_dict
                    assert np.allclose(np.array(output_dict["X"]), model._X)
                    assert np.allclose(np.array(output_dict["y"]), model._y)
                    assert output_dict["eta"] == model._eta
                    assert output_dict["kernel"] == model._kernel.to_dict()
                    assert output_dict["tail"] == model._tail.to_dict()
                    assert np.allclose(np.array(output_dict["lambda"]), model._lambda)
                    assert np.allclose(np.array(output_dict["LU"]), model._LU)
                    assert np.allclose(np.array(output_dict["piv"]), model._piv)
                    assert np.allclose(np.array(output_dict["loo"]), model.loo_residuals)
    
    def test_to_dict_fit_2d(self):
        x = 2*np.pi*np.random.random_sample((20, 2))
        y = func_2d(x)
        eta = 1e-3
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y)
                    output_dict = model.to_dict()
                    assert "X" in output_dict
                    assert "y" in output_dict
                    assert "eta" in output_dict
                    assert "kernel" in output_dict
                    assert "tail" in output_dict
                    assert "lambda" in output_dict
                    assert "LU" in output_dict
                    assert "piv" in output_dict
                    assert "loo" in output_dict
                    assert np.allclose(np.array(output_dict["X"]), model._X)
                    assert np.allclose(np.array(output_dict["y"]), model._y)
                    assert output_dict["eta"] == model._eta
                    assert output_dict["kernel"] == model._kernel.to_dict()
                    assert output_dict["tail"] == model._tail.to_dict()
                    assert np.allclose(np.array(output_dict["lambda"]), model._lambda)
                    assert np.allclose(np.array(output_dict["LU"]), model._LU)
                    assert np.allclose(np.array(output_dict["piv"]), model._piv)
                    assert np.allclose(np.array(output_dict["loo"]), model.loo_residuals)
    
    def test_copy_nofit(self):
        eta = 1
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    copy_model = model.copy()
                    assert copy_model._eta == model._eta
                    assert isinstance(copy_model._kernel, model._kernel.__class__)
                    assert copy_model._kernel.param == model._kernel.param
                    assert isinstance(copy_model._tail, model._tail.__class__)
                    assert np.all(copy_model._tail.params == model._tail.params)
    
    def test_copy_fit_1d(self):
        x = np.linspace(0, 2*np.pi, num = 8)
        y = func_1d(x)
        eta = 1e-3
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y)
                    copy_model = model.copy()
                    assert np.allclose(copy_model._X, model._X)
                    assert np.allclose(copy_model._y, model._y)
                    assert copy_model._eta == model._eta
                    assert isinstance(copy_model._kernel, model._kernel.__class__)
                    assert copy_model._kernel.param == model._kernel.param
                    assert isinstance(copy_model._tail, model._tail.__class__)
                    assert np.all(copy_model._tail.params == model._tail.params)
                    assert np.allclose(copy_model._lambda, model._lambda)
                    assert np.allclose(copy_model._LU, model._LU)
                    assert np.allclose(copy_model._piv, model._piv)
                    assert np.allclose(copy_model.loo_residuals, model.loo_residuals)
    
    def test_copy_fit_2d(self):
        x = 2*np.pi*np.random.random_sample((20, 2))
        y = func_2d(x)
        eta = 1e-3
        for kernel in kernels_dict.values():
            k = kernel(2)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y)
                    copy_model = model.copy()
                    assert np.allclose(copy_model._X, model._X)
                    assert np.allclose(copy_model._y, model._y)
                    assert copy_model._eta == model._eta
                    assert isinstance(copy_model._kernel, model._kernel.__class__)
                    assert copy_model._kernel.param == model._kernel.param
                    assert isinstance(copy_model._tail, model._tail.__class__)
                    assert np.all(copy_model._tail.params == model._tail.params)
                    assert np.allclose(copy_model._lambda, model._lambda)
                    assert np.allclose(copy_model._LU, model._LU)
                    assert np.allclose(copy_model._piv, model._piv)
                    assert np.allclose(copy_model.loo_residuals, model.loo_residuals)
    
    def test_fit_1d_default(self):
        N = 8
        x = np.linspace(0, 2*np.pi, num = N)
        y = func_1d(x)
        eta = 1e-15
        for kernel in kernels_dict.values():
            k = kernel()
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y)
                    assert hasattr(model, "_lambda")
                    assert model._lambda.shape == (N,)
                    assert hasattr(model, "_LU")
                    assert model._LU.shape == (N,N) if t.degree == -1 else (N+1,N+1) if t.degree == 0 else (N+2,N+2)
                    assert hasattr(model, "_piv")
                    assert model._piv.shape == (N,) if t.degree == -1 else (N+1,) if t.degree == 0 else (N+2,)
                    assert hasattr(model, "_loo_residuals")
                    assert model._loo_residuals.shape == (N,)
                    assert model._kernel.param == k.param
                    if t.degree == -1:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))), y.reshape((-1,1)))
                    elif t.degree == 0:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))) + model._tail._p * np.ones((N,1)), y.reshape((-1,1)))
                        assert np.sum(model._lambda) < 1e-5
                        assert np.sum(model._lambda) > -1e-5
                    elif t.degree == 1:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))) 
                            + np.dot(model._tail.compute_P(model._X), np.array(model._tail.params).reshape((-1,1))), y.reshape((-1,1)))
                        assert np.allclose(np.dot(np.transpose(model._tail.compute_P(model._X)), model._lambda.reshape((-1,1))), np.zeros((2,1)))
    
    def test_fit_1d_param(self):
        N = 8
        x = np.linspace(0, 2*np.pi, num = N)
        y = func_1d(x)
        eta = 1e-15
        for kernel in kernels_dict.values():
            k = kernel()
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y, 2)
                    assert hasattr(model, "_lambda")
                    assert model._lambda.shape == (N,)
                    assert hasattr(model, "_LU")
                    assert model._LU.shape == (N,N) if t.degree == -1 else (N+1,N+1) if t.degree == 0 else (N+2,N+2)
                    assert hasattr(model, "_piv")
                    assert model._piv.shape == (N,) if t.degree == -1 else (N+1,) if t.degree == 0 else (N+2,)
                    assert hasattr(model, "_loo_residuals")
                    assert model._loo_residuals.shape == (N,)
                    assert model._kernel.param == 2
                    if t.degree == -1:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))), y.reshape((-1,1)))
                    elif t.degree == 0:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))) + model._tail._p * np.ones((N,1)), y.reshape((-1,1)))
                        assert np.sum(model._lambda) < 1e-5
                        assert np.sum(model._lambda) > -1e-5
                    elif t.degree == 1:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))) 
                            + np.dot(model._tail.compute_P(model._X), np.array(model._tail.params).reshape((-1,1))), y.reshape((-1,1)))
                        assert np.allclose(np.dot(np.transpose(model._tail.compute_P(model._X)), model._lambda.reshape((-1,1))), np.zeros((2,1)))
    
    def test_fit_2d_default(self):
        N = 20
        x = 2*np.pi*np.random.random_sample((N, 2))
        y = func_2d(x)
        eta = 1e-15
        for kernel in kernels_dict.values():
            k = kernel()
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y)
                    assert hasattr(model, "_lambda")
                    assert model._lambda.shape == (N,)
                    assert hasattr(model, "_LU")
                    assert model._LU.shape == (N,N) if t.degree == -1 else (N+1,N+1) if t.degree == 0 else (N+2,N+2)
                    assert hasattr(model, "_piv")
                    assert model._piv.shape == (N,) if t.degree == -1 else (N+1,) if t.degree == 0 else (N+2,)
                    assert hasattr(model, "_loo_residuals")
                    assert model._loo_residuals.shape == (N,)
                    assert model._kernel.param == k.param
                    if t.degree == -1:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))), y.reshape((-1,1)))
                    elif t.degree == 0:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))) + model._tail._p * np.ones((N,1)), y.reshape((-1,1)))
                        assert np.sum(model._lambda) < 1e-5
                        assert np.sum(model._lambda) > -1e-5
                    elif t.degree == 1:
                        assert np.allclose(np.dot(model._Phi(model._X), model._lambda.reshape((-1,1))) 
                            + np.dot(model._tail.compute_P(model._X), np.array(model._tail.params).reshape((-1,1))), y.reshape((-1,1)))
                        assert np.allclose(np.dot(np.transpose(model._tail.compute_P(model._X)), model._lambda.reshape((-1,1))), np.zeros((3,1)))
    
    def test_fit_2d_param(self):
        N = 20
        x = 2*np.pi*np.random.random_sample((N, 2))
        y = func_2d(x)
        eta = 1e-15
        for kernel in kernels_dict.values():
            k = kernel()
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y, 2)
                    assert hasattr(model, "_lambda")
                    assert model._lambda.shape == (N,)
                    assert hasattr(model, "_LU")
                    assert model._LU.shape == (N,N) if t.degree == -1 else (N+1,N+1) if t.degree == 0 else (N+2,N+2)
                    assert hasattr(model, "_piv")
                    assert model._piv.shape == (N,) if t.degree == -1 else (N+1,) if t.degree == 0 else (N+2,)
                    assert hasattr(model, "_loo_residuals")
                    assert model._loo_residuals.shape == (N,)
                    assert model._kernel.param == 2
                    if t.degree == -1:
                        assert np.allclose(np.dot(model._lambda, model._Phi(model._X)), y)
                    elif t.degree == 0:
                        assert np.allclose(np.dot(model._lambda, model._Phi(model._X)) + model._tail._p * np.ones(N), y)
                        assert np.sum(model._lambda) < 1e-5
                        assert np.sum(model._lambda) > -1e-5
                    elif t.degree == 1:
                        assert np.allclose(np.dot(model._lambda, model._Phi(model._X)) + np.dot(np.array(model._tail.params), np.transpose(model._tail.compute_P(model._X))), y)
                        assert np.allclose(np.dot(model._lambda, model._tail.compute_P(model._X)), np.zeros(3))
    
    def test_train_1d(self):
        N = 8
        x = np.linspace(0, 2*np.pi, num = N)
        y = func_1d(x)
        for kernel in kernels_dict.values():
            k = kernel(30)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t)
                    model.train(x, y, method = "Bounded", bounds = [1e-5, 20])
                    assert model._kernel.param >= 1e-5
                    assert model._kernel.param <= 20
                    assert model._is_fitted()
    
    def test_train_2d(self):
        N = 20
        x = 2*np.pi*np.random.random_sample((N, 2))
        y = func_2d(x)
        for kernel in kernels_dict.values():
            k = kernel(30)
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t)
                    model.train(x, y, method = "Bounded", bounds = [1e-5, 20])
                    assert model._kernel.param >= 1e-5
                    assert model._kernel.param <= 20
                    assert model._is_fitted()
    
    def test_predict_nofit(self):
        u = np.linspace(0, 2*np.pi, num = 50)
        for kernel in kernels_dict.values():
            k = kernel()
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t)
                    with pytest.raises(ValueError):
                        p, s = model.predict(u)
    
    def test_predict_1d_fit(self):
        x = np.linspace(0, 2*np.pi, num = 8)
        u = np.linspace(0, 2*np.pi, num = 50)
        y = func_1d(x)
        eta = 1e-15
        for kernel in kernels_dict.values():
            k = kernel()
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y)
                    p, s = model.predict(x)
                    assert p.shape == (len(x),)
                    assert s.shape == (len(x),)
                    assert not np.any(np.isnan(p))
                    assert not np.any(np.isnan(s))
                    assert np.all(s >= 0)
                    assert np.all(s <= 1e-3)
                    assert np.allclose(p, y)
                    p, s = model.predict(u)
                    assert p.shape == (len(u),)
                    assert s.shape == (len(u),)
                    assert not np.any(np.isnan(p))
                    assert not np.any(np.isnan(s))
                    assert np.all(s >= 0)
    
    def test_predict_2d_fit(self):
        x = 2*np.pi*np.random.random_sample((20, 2))
        u = 2*np.pi*np.random.random_sample((50, 2))
        y = func_2d(x)
        eta = 1e-15
        for kernel in kernels_dict.values():
            k = kernel()
            for tail in tails_dict.values():
                t = tail()
                if t.degree >= k.dmin:
                    model = RBF(k, t, eta)
                    model.fit(x, y)
                    p, s = model.predict(x)
                    assert p.shape == (len(x),)
                    assert s.shape == (len(x),)
                    assert not np.any(np.isnan(p))
                    assert not np.any(np.isnan(s))
                    assert np.all(s >= 0)
                    assert np.all(s <= 1e-3)
                    assert np.allclose(p, y)
                    p, s = model.predict(u)
                    assert p.shape == (len(u),)
                    assert s.shape == (len(u),)
                    assert not np.any(np.isnan(p))
                    assert not np.any(np.isnan(s))
                    assert np.all(s >= 0)
    
    def test_sample_y_1d(self):
        pass
    
    def test_sample_y_2d(self):
        pass