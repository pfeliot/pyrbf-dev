import numpy as np
import pytest

from pyrbf.kernels import Kernel, LinearKernel, CubicKernel, ThinPlateKernel, GaussianKernel, MultiQuadraticKernel, InverseQuadraticKernel, InverseMultiQuadraticKernel

class TestKernel:
    def test___init__(self):
        kernel = Kernel(param = 2)
        assert kernel._param == 2
    
    def test___init__default(self):
        kernel = Kernel()
        assert kernel._param == 1
    
    def test_param(self):
        kernel = Kernel(param = 2)
        assert kernel.param == 2
    
    def test_from_dict(self):
        kernels = [Kernel, LinearKernel, CubicKernel, ThinPlateKernel, GaussianKernel, 
            MultiQuadraticKernel, InverseQuadraticKernel, InverseMultiQuadraticKernel]
        for kernel in kernels:
            input_dict = {
                "class" : kernel.__name__,
                "param" : 2
            }
            k = Kernel.from_dict(input_dict)
            assert isinstance(k, kernel)
            assert k.param == 2
    
    def test_to_dict(self):
        kernel = Kernel(1)
        output_dict = kernel.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "Kernel"
        assert "param" in output_dict
        assert output_dict["param"] == 1
    
    def test_copy(self):
        kernel = Kernel(2)
        kernel_copy = kernel.copy()
        assert isinstance(kernel_copy, Kernel)
        assert kernel_copy.param == kernel.param

class TestLinearKernel:
    def test___init__(self):
        kernel = LinearKernel(param = 2)
        assert kernel._param == 2
    
    def test___init__default(self):
        kernel = LinearKernel()
        assert kernel._param == 1
    
    def test_dmin(self):
        kernel = LinearKernel(1)
        assert kernel.dmin == 0
        
    def test_to_dict(self):
        kernel = LinearKernel(1)
        output_dict = kernel.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "LinearKernel"
        assert "param" in output_dict
        assert output_dict["param"] == 1
        
    def test_evaluate_1(self):
        kernel = LinearKernel(1)
        assert kernel.evaluate(0) == 0
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == D)
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == D)
        
    def test_evaluate_2(self):
        kernel = LinearKernel(2)
        assert kernel.evaluate(0) == 0
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == 2*D)
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == 2*D)
    
    def test_derivatives_1(self):
        kernel = LinearKernel(1)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == np.ones_like(D))
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == np.ones_like(D))
        
    def test_derivatives_2(self):
        kernel = LinearKernel(2)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == 2 * np.ones_like(D))
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == 2 * np.ones_like(D))

class TestCubicKernel:
    def test___init__(self):
        kernel = CubicKernel(param = 2)
        assert kernel._param == 2
    
    def test___init__default(self):
        kernel = CubicKernel()
        assert kernel._param == 1
    
    def test_dmin(self):
        kernel = CubicKernel(1)
        assert kernel.dmin == 1
        
    def test_to_dict(self):
        kernel = CubicKernel(1)
        output_dict = kernel.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "CubicKernel"
        assert "param" in output_dict
        assert output_dict["param"] == 1
        
    def test_evaluate_1(self):
        kernel = CubicKernel(1)
        assert kernel.evaluate(0) == 0
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == D**3)
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == D**3)
        
    def test_evaluate_2(self):
        kernel = CubicKernel(2)
        assert kernel.evaluate(0) == 0
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == 2*D**3)
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == 2*D**3)
    
    def test_derivatives_1(self):
        kernel = CubicKernel(1)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == 2*D**2)
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == 2*D**2)
        
    def test_derivatives_2(self):
        kernel = CubicKernel(2)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == 4*D**2)
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == 4*D**2)

class TestThinPlateKernel:
    def test___init__(self):
        kernel = ThinPlateKernel(param = 2)
        assert kernel._param == 2
    
    def test___init__default(self):
        kernel = ThinPlateKernel()
        assert kernel._param == 1
    
    def test_dmin(self):
        kernel = ThinPlateKernel(1)
        assert kernel.dmin == 1
        
    def test_to_dict(self):
        kernel = ThinPlateKernel(1)
        output_dict = kernel.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "ThinPlateKernel"
        assert "param" in output_dict
        assert output_dict["param"] == 1
        
    def test_evaluate_1(self):
        kernel = ThinPlateKernel(1)
        assert kernel.evaluate(np.array([0])) == 0
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == D**2*np.log(D))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == D**2*np.log(D))
        
    def test_evaluate_2(self):
        kernel = ThinPlateKernel(2)
        assert kernel.evaluate(np.array([0])) == 0
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == D**2*np.log(2*D))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == D**2*np.log(2*D))
    
    def test_derivatives_1(self):
        kernel = ThinPlateKernel(1)
        D = np.random.rand(10)
        assert np.allclose(kernel.derivatives(D), D + 2*D*np.log(D))
        D = np.random.rand(10,2)
        assert np.allclose(kernel.derivatives(D), D + 2*D*np.log(D))
        
    def test_derivatives_2(self):
        kernel = ThinPlateKernel(2)
        D = np.random.rand(10)
        assert np.allclose(kernel.derivatives(D), D + 2*D*np.log(2*D))
        D = np.random.rand(10,2)
        assert np.allclose(kernel.derivatives(D), D + 2*D*np.log(2*D))

class TestGaussianKernel:
    def test___init__(self):
        kernel = GaussianKernel(param = 2)
        assert kernel._param == 2
    
    def test___init__default(self):
        kernel = GaussianKernel()
        assert kernel._param == 1
    
    def test_dmin(self):
        kernel = GaussianKernel(1)
        assert kernel.dmin == -1
        
    def test_to_dict(self):
        kernel = GaussianKernel(1)
        output_dict = kernel.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "GaussianKernel"
        assert "param" in output_dict
        assert output_dict["param"] == 1
        
    def test_evaluate_1(self):
        kernel = GaussianKernel(1)
        assert kernel.evaluate(np.array([0])) == 1
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == np.exp(-D**2))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == np.exp(-D**2))
        
    def test_evaluate_2(self):
        kernel = GaussianKernel(2)
        assert kernel.evaluate(np.array([0])) == 1
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == np.exp(-D**2/4))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == np.exp(-D**2/4))
    
    def test_derivatives_1(self):
        kernel = GaussianKernel(1)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == -2*D*np.exp(-D**2))
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == -2*D*np.exp(-D**2))
        
    def test_derivatives_2(self):
        kernel = GaussianKernel(2)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == -D/2*np.exp(-D**2/4))
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == -D/2*np.exp(-D**2/4))

class TestMultiQuadraticKernel:
    def test___init__(self):
        kernel = MultiQuadraticKernel(param = 2)
        assert kernel._param == 2
    
    def test___init__default(self):
        kernel = MultiQuadraticKernel()
        assert kernel._param == 1
    
    def test_dmin(self):
        kernel = MultiQuadraticKernel(1)
        assert kernel.dmin == 0
        
    def test_to_dict(self):
        kernel = MultiQuadraticKernel(1)
        output_dict = kernel.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "MultiQuadraticKernel"
        assert "param" in output_dict
        assert output_dict["param"] == 1
        
    def test_evaluate_1(self):
        kernel = MultiQuadraticKernel(1)
        assert kernel.evaluate(np.array([0])) == 1
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == np.sqrt(D**2 + np.ones_like(D)))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == np.sqrt(D**2 + np.ones_like(D)))
        
    def test_evaluate_2(self):
        kernel = MultiQuadraticKernel(2)
        assert kernel.evaluate(np.array([0])) == 1
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == np.sqrt(4*D**2 + np.ones_like(D)))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == np.sqrt(4*D**2 + np.ones_like(D)))
    
    def test_derivatives_1(self):
        kernel = MultiQuadraticKernel(1)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == D/np.sqrt(D**2 + np.ones_like(D)))
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == D/np.sqrt(D**2 + np.ones_like(D)))
        
    def test_derivatives_2(self):
        kernel = MultiQuadraticKernel(2)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == 4*D/np.sqrt(4*D**2 + np.ones_like(D)))
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == 4*D/np.sqrt(4*D**2 + np.ones_like(D)))

class TestInverseQuadraticKernel:
    def test___init__(self):
        kernel = InverseQuadraticKernel(param = 2)
        assert kernel._param == 2
    
    def test___init__default(self):
        kernel = InverseQuadraticKernel()
        assert kernel._param == 1
    
    def test_dmin(self):
        kernel = InverseQuadraticKernel(1)
        assert kernel.dmin == -1
        
    def test_to_dict(self):
        kernel = InverseQuadraticKernel(1)
        output_dict = kernel.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "InverseQuadraticKernel"
        assert "param" in output_dict
        assert output_dict["param"] == 1
        
    def test_evaluate_1(self):
        kernel = InverseQuadraticKernel(1)
        assert kernel.evaluate(np.array([0])) == 1
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == (D**2 + np.ones_like(D))**(-1))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == (D**2 + np.ones_like(D))**(-1))
        
    def test_evaluate_2(self):
        kernel = InverseQuadraticKernel(2)
        assert kernel.evaluate(np.array([0])) == 1
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == (4*D**2 + np.ones_like(D))**(-1))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == (4*D**2 + np.ones_like(D))**(-1))
    
    def test_derivatives_1(self):
        kernel = InverseQuadraticKernel(1)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == -2*D/(D**2 + np.ones_like(D))**2)
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == -2*D/(D**2 + np.ones_like(D))**2)
        
    def test_derivatives_2(self):
        kernel = InverseQuadraticKernel(2)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == -8*D/(4*D**2 + np.ones_like(D))**2)
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == -8*D/(4*D**2 + np.ones_like(D))**2)

class TestInverseMultiQuadraticKernel:
    def test___init__(self):
        kernel = InverseMultiQuadraticKernel(param = 2)
        assert kernel._param == 2
    
    def test___init__default(self):
        kernel = InverseMultiQuadraticKernel()
        assert kernel._param == 1
    
    def test_dmin(self):
        kernel = InverseMultiQuadraticKernel(1)
        assert kernel.dmin == -1
        
    def test_to_dict(self):
        kernel = InverseMultiQuadraticKernel(1)
        output_dict = kernel.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "InverseMultiQuadraticKernel"
        assert "param" in output_dict
        assert output_dict["param"] == 1
        
    def test_evaluate_1(self):
        kernel = InverseMultiQuadraticKernel(1)
        assert kernel.evaluate(np.array([0])) == 1
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == np.sqrt(D**2 + np.ones_like(D))**(-1))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == np.sqrt(D**2 + np.ones_like(D))**(-1))
        
    def test_evaluate_2(self):
        kernel = InverseMultiQuadraticKernel(2)
        assert kernel.evaluate(np.array([0])) == 1
        D = np.random.rand(10)
        assert np.all(kernel.evaluate(D) == np.sqrt(4*D**2 + np.ones_like(D))**(-1))
        D = np.random.rand(10,2)
        assert np.all(kernel.evaluate(D) == np.sqrt(4*D**2 + np.ones_like(D))**(-1))
    
    def test_derivatives_1(self):
        kernel = InverseMultiQuadraticKernel(1)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == -D/np.sqrt(D**2 + np.ones_like(D))**3)
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == -D/np.sqrt(D**2 + np.ones_like(D))**3)
        
    def test_derivatives_2(self):
        kernel = InverseMultiQuadraticKernel(2)
        D = np.random.rand(10)
        assert np.all(kernel.derivatives(D) == -4*D/np.sqrt(4*D**2 + np.ones_like(D))**3)
        D = np.random.rand(10,2)
        assert np.all(kernel.derivatives(D) == -4*D/np.sqrt(4*D**2 + np.ones_like(D))**3)