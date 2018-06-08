import numpy as np
from Decorator import clock

class Solver:

    @clock
    def gradient_descent_mtd(self, f, x):    
        while True:
            neg_grad = - f.grad(x)
            opt_alpha = self.bisection(f, x, neg_grad)
            x_new = x + opt_alpha * neg_grad
            if self.__is_conv(x, x_new, f= f):
                break
            else:
                x = x_new
        return x

    def bisection(self, f, x, direc, s= 1):
        '''phi must be in C^1, and phi prime (0) must be less than 0
        '''
        # Check if \phi \prime (0) < 0
        if np.dot(direc, f.grad(x)) >= 0:
            raise ValueError("The initial point must be a descent direction!")

        # Find initial interval
        alpha_min = 0

        while True:
            x_new = x + s * direc
            phi_prime = np.dot(direc, f.grad(x_new))
            if phi_prime > 0:
                alpha_max = s
                break
            else:
                alpha_min = s
                s = 2 * s 

        # Main part
        while True:
            alpha = (alpha_min + alpha_max) / 2
            x_new = x + alpha * direc
            if np.dot(direc, f.grad(x_new)) > 0:
                alpha_max = alpha
            else:
                alpha_min = alpha
            if self.__is_conv(alpha_max, alpha_min):
                break
        
        return (alpha_max + alpha_min)/2

    def __is_conv(self, x1, x2, f= lambda x:x, accuracy= 1e-6):
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        if np.linalg.norm(f(x1) - f(x2)) < accuracy:
            return True
        else:
            return False
        
        
