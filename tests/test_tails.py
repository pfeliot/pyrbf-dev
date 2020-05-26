import numpy as np
import pytest

from pyrbf.tails import Tail, NoTail, ConstantTail, LinearTail

class TestTail:
    def test___init__(self):
        tail = Tail()
        assert True
        
    def test_params(self):
        tail = Tail()
        assert tail.params is None
        
    def test_set_params(self):
        tail = Tail()
        tail.set_params(1)
        assert True
    
    def test_from_dict(self):
        tails = [Tail, NoTail, ConstantTail, LinearTail]
        for tail in tails:
            t_ = tail()
            input_dict = {
                "class" : tail.__name__,
                "params" : t_.params
            }
            t = Tail.from_dict(input_dict)
            assert isinstance(t, tail)
            assert t.params == t_.params
    
    def test_to_dict(self):
        tail = Tail()
        output_dict = tail.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "Tail"
        assert "params" in output_dict
        assert output_dict["params"] is None
    
    def test_copy(self):
        tail = Tail()
        tail_copy = tail.copy()
        assert isinstance(tail_copy, Tail)
        assert tail_copy.params is None

class TestNoTail:
    def test___init__(self):
        tail = NoTail()
        assert True
        
    def test_params(self):
        tail = NoTail()
        assert tail.params is None
        
    def test_set_params(self):
        tail = NoTail()
        tail.set_params(1)
        assert True
    
    def test_degree(self):
        tail = NoTail()
        assert tail.degree == -1
        
    def test_to_dict(self):
        tail = NoTail()
        output_dict = tail.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "NoTail"
        assert "params" in output_dict
        assert output_dict["params"] is None
        
    def test_evaluate(self):
        tail = NoTail()
        x = np.random.rand(10, 2)
        y = tail.evaluate(x)
        assert y.shape == (10,)
        assert np.all(y == np.zeros(10))
        x = np.random.rand(20, 2)
        y = tail.evaluate(x)
        assert y.shape == (20,)
        assert np.all(y == np.zeros(20))
        
    def test_compute_b(self):
        tail = NoTail()
        y = np.random.rand(10)
        b = tail.compute_b(y)
        assert b.shape == (10,1)
        assert np.all(b == y.reshape((-1,1)))
        b = tail.compute_b(y, 2)
        assert b.shape == (10,1)
        assert np.all(b == y.reshape((-1,1)))
        
    def test_compute_P(self):
        tail = NoTail()
        x = np.random.rand(10, 2)
        P = tail.compute_P(x)
        assert P is None

class TestConstantTail:
    def test___init__(self):
        tail = ConstantTail()
        assert True
        
    def test_params(self):
        tail = ConstantTail()
        assert tail.params == 0
        
    def test_set_params(self):
        tail = ConstantTail()
        tail.set_params(2)
        assert tail.params == 2
    
    def test_degree(self):
        tail = ConstantTail()
        assert tail.degree == 0
        
    def test_to_dict(self):
        tail = ConstantTail()
        tail.set_params(2)
        output_dict = tail.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "ConstantTail"
        assert "params" in output_dict
        assert output_dict["params"] == 2
        
    def test_evaluate(self):
        tail = ConstantTail()
        tail.set_params(2)
        x = np.random.rand(10, 2)
        y = tail.evaluate(x)
        assert y.shape == (10,)
        assert np.all(y == 2*np.ones(10))
        tail.set_params(0)
        x = np.random.rand(20, 2)
        y = tail.evaluate(x)
        assert y.shape == (20,)
        assert np.all(y == np.zeros(20))
        
    def test_compute_b(self):
        tail = ConstantTail()
        y = np.random.rand(10)
        b = tail.compute_b(y)
        assert b.shape == (11,1)
        z = np.zeros(11)
        z[:10] = y
        assert np.all(b == z.reshape((-1,1)))
        b = tail.compute_b(y, 3)
        assert b.shape == (11,1)
        z = np.zeros(11)
        z[:10] = y
        assert np.all(b == z.reshape((-1,1)))
        
    def test_compute_P(self):
        tail = ConstantTail()
        x = np.random.rand(10, 2)
        P = tail.compute_P(x)
        assert np.all(P == np.ones((10,1)))

class TestLinearTail:
    def test___init__(self):
        tail = LinearTail()
        assert True
        
    def test_params(self):
        tail = LinearTail()
        assert tail.params == [0, 0]
        
    def test_set_params(self):
        tail = LinearTail()
        tail.set_params([1,1,1])
        assert tail.params == [1, 1, 1]
        assert np.all(tail._p == np.array([1, 1]))
        assert tail._c == 1
    
    def test_degree(self):
        tail = LinearTail()
        assert tail.degree == 1
        
    def test_to_dict(self):
        tail = LinearTail()
        tail.set_params([1,1,1])
        output_dict = tail.to_dict()
        assert "class" in output_dict
        assert output_dict["class"] == "LinearTail"
        assert "params" in output_dict
        assert len(output_dict["params"]) == 3
        assert output_dict["params"] == [1, 1, 1]
        
    def test_evaluate(self):
        tail = LinearTail()
        tail.set_params([1,1,1])
        x = np.random.rand(10, 2)
        y = tail.evaluate(x)
        assert y.shape == (10,)
        assert np.allclose(y, x[:,0].ravel() + x[:,1].ravel() + np.ones(10))
        tail.set_params([0,0,0])
        x = np.random.rand(20, 2)
        y = tail.evaluate(x)
        assert y.shape == (20,)
        assert np.all(y == np.zeros(20))
        
    def test_compute_b_1d(self):
        tail = LinearTail()
        y = np.random.rand(10)
        b = tail.compute_b(y, d = 1)
        assert b.shape == (12,1)
        z = np.zeros(12)
        z[:10] = y
        assert np.all(b == z.reshape((-1,1)))
        
    def test_compute_b_2d(self):
        tail = LinearTail()
        y = np.random.rand(10)
        b = tail.compute_b(y, d = 2)
        assert b.shape == (13,1)
        z = np.zeros(13)
        z[:10] = y
        assert np.all(b == z.reshape((-1,1)))
        
    def test_compute_P_1d(self):
        tail = LinearTail()
        x = np.random.rand(10, 1)
        P = tail.compute_P(x)
        assert np.all(P == np.hstack((x, np.ones((10,1)))))
        
    def test_compute_P_2d(self):
        tail = LinearTail()
        x = np.random.rand(10, 2)
        P = tail.compute_P(x)
        assert np.all(P == np.hstack((x, np.ones((10,1)))))