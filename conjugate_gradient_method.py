import numpy as np
import basic_algorithms as ba

def conjugateGradientMethod(func, X, displayDetail= False, decimal= 4, accuracy= 1e-6):

    counter = 0

    negativeGrad = - ba.numerical_differentiate.Grad(func, X)
    conjugateDirection = negativeGrad

    if displayDetail == True:
        print("INITIALIZING... ")
        print("CONJUGATE DIRECTION (NEG-GRADIENT DIRECTION) AT POINT" + str(X.round(decimal))
              + " IS " + str(conjugateDirection.round(decimal)))

    optAlpha = ba.linear_search.GoldenSectionMethod(func, X, conjugateDirection, accuracy)
    X = X + optAlpha * conjugateDirection

    counter += 1

    if displayDetail == True:
        print("INITIALIZATION IS FINISHED, ITERATION BEGINS...")
        ba.decoration.plotDashLine()

    while True:

        if displayDetail == True:
            print("ITERATION TIMES: " + str(counter))

        formerNegativeGrad = negativeGrad
        currentNegativeGrad = - ba.numerical_differentiate.Grad(func, X)

        if displayDetail == True:
            print("NEG-GRADIENT AT POINT " + str(X.round(decimal))
                  + " IS " + str(currentNegativeGrad.round(decimal)))

        if ba.check_condition.isGradientConvergent2Zero(currentNegativeGrad):
            if displayDetail == True:
                print("ITERATION BREAK! THE MINIMAL VALUE OBTAINED AT POINT: " + str(X.round(decimal)))
                ba.decoration.plotDashLine()
            break

        conjugateDirection = currentNegativeGrad + (currentNegativeGrad.dot(currentNegativeGrad)) / (formerNegativeGrad.dot(formerNegativeGrad)) * conjugateDirection

        if displayDetail == True:
            print("CONJUGATE DIRECTION AT POINT " + str(X.round(decimal))
                  + " IS " + str(conjugateDirection.round(decimal)))

        optAlpha = ba.linear_search.GoldenSectionMethod(func, X, conjugateDirection, accuracy)

        X = X + optAlpha * conjugateDirection

        counter += 1
        ba.decoration.plotDashLine()


# ===============================================

def myFunc(x):
    return 0.5*x[0]*x[0] + x[1]*x[1]
    # return 2*(x[0]+2.3)**2 + (x[1]-1.5)**2

initialX = np.array([2, 1], dtype= float)

optX = conjugateGradientMethod(myFunc, initialX, displayDetail= True)

