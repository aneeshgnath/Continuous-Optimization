import numpy as np

def myFunc(x):
    return 2*(x[0]*+2.3)**2 + (x[1]-1.5)**2


a = np.array([1,2,4])
b = a.copy()
b[2] = 31
print(a, b)
