import numpy as np

class FuncObj:
    ''' It is deserved to write Function Object, because many mathematical properties must be discussed in the optimization problem, just like continuity, which are not considered in the python function class. However, discuss some mathematical properties like open and close is useless in computer programming, but I will try my best.
    '''
    def __init__(self, f= lambda : None, var_num= 0, display_dicimals= 5):
        self._f = f
        self._var_num = var_num

    def grad(self, x, step_len= 1e-6):
        """Compute the gradient of a function at a given point
        
        Parameters
        ----------
            func : A function object
            x : An array-like point represented where the gradient was computed
            step_len : The scale of difference we used in the computation of gradient, when the step length tends to 0 we get the true gradient

        Return
        ------
            gradient : An array containing the gradient
        """

        import operator as op
        x = np.asarray(x, dtype= float)

        grad = [ (self._f(self.__bit_op(x, i, step_len, op.add)) -
            self._f(self.__bit_op(x, i, step_len, op.sub))) / (2 * step_len)
            for i in range(len(x))]
   
        return grad

    def Hessian(self, x, step_len= 1e-6):
        
        import operator as op
        x = np.asarray(x, dtype= float)
        
        hessian = np.zeros((x.size, x.size))

        for i in range(x.size):
            for j in range(x.size):
                if i == j:
                    p1 = self.__bit_op(x, i, step_len, op.sub)
                    p2 = self.__bit_op(x, i, step_len, op.add)
                    hessian[i][j] = (self._f(p1) + self._f(p2) - 2 * self._f(x)) / step_len ** 2
                else:
                    p1 = self.__2bit_op(x, i, step_len, op.add, j, step_len, op.add)
                    p2 = self.__2bit_op(x, i, step_len, op.sub, j, step_len, op.sub)
                    p3 = self.__bit_op(x, i, step_len, op.add)
                    p4 = self.__bit_op(x, i, step_len, op.sub)
                    p5 = self.__bit_op(x, j, step_len, op.add)
                    p6 = self.__bit_op(x, j, step_len, op.sub)
                    hessian[i][j] = (self._f(p1) + self._f(p2) - self._f(p3) - 
                        self._f(p4) - self._f(p5) - self._f(p6) + 
                        2 * self._f(x)) / (2 * step_len ** 2)
                        
        return hessian

    def __bit_op(self, x, pos, a, op):
        y = x.copy()
        y[pos] = op(y[pos], a)
        return y
    
    def __2bit_op(self, x, pos1, a1, op1, pos2, a2, op2):
        y = x.copy()
        y[pos1] = op1(y[pos1], a1)
        y[pos2] = op2(y[pos2], a2)
        return y

        
if __name__ == "__main__":
    
    def f(x):
        return np.sin(x[0]) + x[1]*x[1]*x[0]

    myFunc = FuncObj(f= f, var_num= 2)
    x= [2, 1]
    print(myFunc.grad(x))
    print(myFunc.Hessian(x))
