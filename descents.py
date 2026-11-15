import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p
       


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta
    
    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        self.model.loss_history = []
        
        for _ in range(self.max_iter):
            
            self.model.loss_history.append(self.model.compute_loss())
            
            delta = self._step()
            
            if np.isnan(delta).any():
                break
                
            if np.sum(delta ** 2) < self.tolerance:
                break
                
        self.model.loss_history.append(self.model.compute_loss())


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:

        gradient = self.model.compute_gradients()
        
        lr = self.lr_schedule.get_lr(self.iteration)
        
        delta = -lr * gradient
        
        self.model.w += delta
        
        return delta

class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # 1) выбрать случайный батч
        n_samples = self.model.X_train.shape[0]
        indices = np.random.choice(n_samples, self.batch_size, replace=False)
        X_batch = self.model.X_train[indices]
        y_batch = self.model.y_train[indices]
        
        # 2) вычислить градиенты на батче
        gradient = self.model.compute_gradients(X_batch, y_batch)
        
        # 3) обновить веса модели
        lr = self.lr_schedule.get_lr(self.iteration)
        delta = -lr * gradient
        self.model.w += delta
        
        return delta
        


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.avg_grad = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.avg_grad = np.zeros(num_features)

        actual_batch_size = min(self.batch_size, num_objects)
        indices = np.random.choice(num_objects, actual_batch_size, replace=False)

        for idx in indices:
            X_i = X_train[idx:idx+1]
            y_i = y_train[idx:idx+1]
            
            new_grad = self.model.compute_gradients(X_i, y_i)
            
            old_grad = self.grad_memory[idx]
            self.avg_grad += (new_grad - old_grad) / num_objects
            
            self.grad_memory[idx] = new_grad

        lr = self.lr_schedule.get_lr(self.iteration)
        delta = -lr * self.avg_grad
        self.model.w += delta
        
        return delta

class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
            
        gradient = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)
        
        self.velocity = self.beta * self.velocity + lr * gradient
        
        delta = -self.velocity
        self.model.w += delta
        
        return delta


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)
            
        gradient = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)
        
        k = self.iteration + 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** k)
        v_hat = self.v / (1 - self.beta2 ** k)
        
        delta = -lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        self.model.w += delta
        
        return delta


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        self.model.w = self.model.loss_function.analytic_solution(
            self.model.X_train, self.model.y_train
        )
