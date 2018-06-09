import numpy as np
from numpy import linalg as LA
from Decorator import clock

class Solver:

    @clock
    def gradient_descent_mtd(self, f, x, accuracy = 1e-4):    
        '''The object function must be in C^1
        '''
        while True:
            neg_grad = - f.grad(x)
            opt_alpha = self.bisection(f, x, neg_grad)
            x_new = x + opt_alpha * neg_grad
            if self.__is_conv(x, x_new, f= f, accuracy= accuracy):
                break
            else:
                x = x_new
        return x_new

    @clock
    def newton_mtd(self, f, x, accuracy= 1e-4):
        '''The object function must be in C^1, and pay attention to the following points: 
        1. At some points the Hessian matrix might be singular, which will lead to the failure of newton method 
        2. In some situations the newton method will converges into a stationary point.
        3. Newton method is only locally convergent, which means if the initial point is not close enough to the global optimal point, assuming it exsits, the algorithm may not converge.
        '''
        while True:
            H = f.hessian(x)
            grad = f.grad(x)
            x_new = x - np.matmul(LA.inv(H), grad)
            if self.__is_conv(x, x_new, f= f, accuracy= accuracy):
                break
            else:
                x = x_new
        return x_new

    def bisection(self, f, x, direc, accuracy= 1e-6, s= 1, threshold= 1e5):
        '''Phi must be in C^1, and phi'(0) must be less than 0
        '''

        if np.dot(direc, f.grad(x)) >= 0:
            raise ValueError("The initial point must be a descent direction!")

        alpha_min = 0

        while True:

            if s - alpha_min > threshold:
                raise ValueError("\n Length of search interval = %d \n Check if this unconstrained problem turely has the minimal value (not equal to minus infinity), and if it has, enlarge the default parameter `threshold = %.1e` of bisection method and retry." % (s-alpha_min, threshold))

            x_new = x + s * direc
            phi_prime = np.dot(direc, f.grad(x_new))
            if phi_prime > 0:
                alpha_max = s
                break
            else:
                alpha_min = s
                s = 2 * s 

        while True:
            alpha = (alpha_min + alpha_max) / 2
            x_new = x + alpha * direc
            if np.dot(direc, f.grad(x_new)) > 0:
                alpha_max = alpha
            else:
                alpha_min = alpha
            if self.__is_conv(alpha_max, alpha_min, accuracy= accuracy):
                break
        
        return (alpha_max + alpha_min)/2

    def __is_conv(self, x1, x2, f= lambda x:x, accuracy= 1e-6):
        '''Function for checking the convergence of two points.
        
        Parameters
        ----------
        x1 : Array-like object, the first point in R^n
        x2 : Array-like object, the second point in R^n
        f : Function (with range in R^m) used to transfrom x1 x2 (in R^n) into R^m, then we can discuss the convergence after the transformation, i.e., for given small epsilon (the parameter accuracy), we wonder know if the inequality `|| f(x1) - f(x2) || < epsilon` holds. In the case that no specific function is input, an identical function will be activated.
        accuracy : It is the aforementioned epsilon

        Return
        ------
        Boolean value which indicates if two points are already convergent under given parameters
        '''
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        if np.linalg.norm(f(x1) - f(x2)) < accuracy:
            return True
        else:
            return False
        
        
