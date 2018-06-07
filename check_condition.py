import numpy as np

def isConvergent(x, newX, accuracy= 1e-5):

    difference = np.linalg.norm(x - newX)/np.linalg.norm(x)

    if difference <= accuracy:
        return True
    else:
        return False

def isNewtonConvergent(gradient, invHessianMatrix, accuracy= 1e-5):

    val = np.sqrt(gradient.dot(invHessianMatrix).dot(gradient))

    if val/2. < accuracy:
        return True
    else:
        return False

def isGradientConvergent2Zero(gradient, accuracy= 1e-5):

    if np.linalg.norm(gradient) < accuracy:
        return True
    else:
        return False

def isPositiveDenifite(hessianMatrix):

    eigenvalues = np.linalg.eig(hessianMatrix)[0]

    for eigenvalue in eigenvalues:
        if eigenvalue <= 0:
            return  False

    return True