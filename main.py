from Func import FuncObj
from Mod import Model
from Sol import Solver
import numpy as np

if __name__ == "__main__":
    
    def f(x):
        # ROSENBROCK FUNCTION
        return 100*(x[1]-x[0]**2)**2 + (1-x[0]**2)
    
    myFunc = FuncObj(f= f, var_num= 2)
    x= [2, 1]

    solver = Solver()
    print(solver.gradient_descent_mtd(myFunc, x))
    print(solver.newton_mtd(myFunc, x))
