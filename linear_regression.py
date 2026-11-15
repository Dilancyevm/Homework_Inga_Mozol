import numpy as np 
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC
from scipy.sparse.linalg import svds



class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        l = X.shape[0]
        y_pred = X.dot(w)
        return np.mean((y_pred - y) ** 2)
    

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        l = X.shape[0]
        y_pred = X.dot(w)
        return (2 / l) * X.T.dot(y_pred - y)
    

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса; 
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать

        return self.analytic_solution_func(X, y)
        
    
    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        X_transpose = X.T
        
        XTX = X_transpose.dot(X)
        
        XTX_inv = np.linalg.pinv(XTX)
        
        w = XTX_inv.dot(X_transpose).dot(y)
        
        return w
    
    
    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        """
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        S_inv = np.zeros_like(S)
        nonzero = S > 1e-12
        S_inv[nonzero] = 1.0 / S[nonzero]
        return Vt.T.dot(np.diag(S_inv)).dot(U.T).dot(y)
        """
        k = min(X.shape) - 1
        
        U, s, Vt = svds(X, k=k, tol=0)
        
        s_inv = np.zeros_like(s)
        
        threshold = 1e-12
        nonzero = s > threshold
        s_inv[nonzero] = 1.0 / s[nonzero]
        
        w = Vt.T.dot(np.diag(s_inv)).dot(U.T).dot(y)
        
        return w
    


class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate
    
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        core_loss_val = self.core_loss.loss(X, y, w)
        penalty_val = (self.mu_rate / 2) * np.sum(w ** 2)
        return core_loss_val + penalty_val

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)
        
        penalty_part = self.mu_rate * w
        
        return core_part + penalty_part



class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        return X.dot(self.w)

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is None or y_batch is None:
            return self.loss_function.gradient(self.X_train, self.y_train, self.w)
        return self.loss_function.gradient(X_batch, y_batch, self.w)

    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is None or y_batch is None:
            return self.loss_function.loss(self.X_train, self.y_train, self.w)
        return self.loss_function.loss(X_batch, y_batch, self.w)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        self.X_train, self.y_train = X, y
        
        self.w = np.zeros(X.shape[1])
        
        self.optimizer.optimize()

class LogCoshLoss(LossFunction):
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        a = X.dot(w)
        return np.mean(np.log(np.cosh(a - y)))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        l = X.shape[0]
        a = X.dot(w)
        return X.T.dot(np.tanh(a - y)) / l


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        a = X.dot(w)
        error = a - y
        abs_error = np.abs(error)
        
        mask = abs_error < self.delta
        
        loss_val = np.zeros_like(error, dtype=float)
        loss_val[mask] = 0.5 * (error[mask] ** 2)
        loss_val[~mask] = self.delta * abs_error[~mask] - 0.5 * (self.delta ** 2)
        
        return np.mean(loss_val)

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        l = X.shape[0]
        a = X.dot(w)
        error = a - y
        
        grad_vec = np.zeros_like(error)
        mask = np.abs(error) < self.delta
        grad_vec[mask] = error[mask]
        grad_vec[~mask] = self.delta * np.sign(error[~mask])
        
        return X.T.dot(grad_vec) / l
