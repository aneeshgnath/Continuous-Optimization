from Func import FuncObj
from Mod import Model
from Sol import Solver
import numpy as np

if __name__ == "__main__":
    
    def f(x):
        return np.sin(x[0]) + x[1]*x[1]*x[0]
    
    myFunc = FuncObj(f= f, var_num= 2)
    x= [2, 1]

    solver = Solver()
    opt = solver.gradient_descent_mtd(f= myFunc, x= x)
    print("Optimal: ", opt, myFunc(opt))

