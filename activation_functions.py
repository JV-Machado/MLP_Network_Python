from abc import ABC, abstractmethod
import math

class ActivationFunction(ABC):

    @staticmethod    
    @abstractmethod
    def g(u):
        pass
    
class DerivativeFunction(ABC):
    
    @staticmethod    
    @abstractmethod
    def dg(u):
        pass
    
    
class BinaryStep(ActivationFunction):
    
    def g(u):
        return 1 if u >= 0 else 0
    
    
class SignFunction(ActivationFunction):
    
    def g(u):
        return 1 if u >= 0 else -1
    
        
class LogisticFunction(ActivationFunction):
    
    def g(u):
        return (1/(1 + math.e**-u))
    
class LogisticDerivativeFunction(DerivativeFunction):
    
    def dg(u):
        return (LogisticFunction.g(u)) * (1 - LogisticFunction.g(u))
        
        
        
        
        
        
        
        
        
        
        
        
        
        