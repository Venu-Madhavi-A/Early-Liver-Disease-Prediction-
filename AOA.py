import time
from math import inf

import numpy as np
from mpmath import eps


def AOA(X,F_obj, LB, UB, M_Iter ):
    N,Dim = X.shape[0],X.shape[1]
    # Two variables to keep the positions and the fitness value of the best-obtained solution
    Best_P = np.zeros((1, Dim))
    Best_FF = inf
    Conv_curve = np.zeros((1, M_Iter))
    # Initialize the positions of solution
    Xnew = X
    Ffun = np.zeros((1, X.shape[1 - 1]))
    Ffun_new = np.zeros((1, Xnew.shape[1 - 1]))
    MOP_Max = 1
    MOP_Min = 0.2
    C_Iter = 1
    Alpha = 5
    Mu = 0.499
    for i in np.arange(1, X.shape[1 - 1] + 1).reshape(-1):
        Ffun[1, i] = F_obj(X[i,:])
        if Ffun[1, i] < Best_FF:
            Best_FF = Ffun[1, i]
            Best_P = X[i,:]
    ct = time.time()
    while C_Iter < M_Iter + 1:
        MOP = 1 - ((C_Iter) ** (1 / Alpha) / (M_Iter) ** (1 / Alpha))
        MOA = MOP_Min + C_Iter * ((MOP_Max - MOP_Min) / M_Iter)
        # Update the Position of solutions
        for i in np.arange(1, X.shape[1 - 1] + 1).reshape(-1):
            for j in np.arange(1, X.shape[2 - 1] + 1).reshape(-1):
                r1 = np.random.rand()
                if (LB.shape[2 - 1] == 1):
                    if r1 < MOA:
                        r2 = np.random.rand()
                        if r2 > 0.5:
                            Xnew[i, j] = Best_P[1, j] / (MOP + eps) * ((UB - LB) * Mu + LB)
                        else:
                            Xnew[i, j] = Best_P[1, j] * MOP * ((UB - LB) * Mu + LB)
                    else:
                        r3 = np.random.rand()
                        if r3 > 0.5:
                            Xnew[i, j] = Best_P[1, j] - MOP * ((UB - LB) * Mu + LB)
                        else:
                            Xnew[i, j] = Best_P[1, j] + MOP * ((UB - LB) * Mu + LB)
                if (LB.shape[2 - 1] != 1):
                    r1 = np.random.rand()
                    if r1 < MOA:
                        r2 = np.random.rand()
                        if r2 > 0.5:
                            Xnew[i, j] = Best_P[1, j] / (MOP + eps) * ((UB(j) - LB(j)) * Mu + LB(j))
                        else:
                            Xnew[i, j] = Best_P[1, j]* MOP * ((UB(j) - LB(j)) * Mu + LB(j))
                    else:
                        r3 = np.random.rand()
                        if r3 > 0.5:
                            Xnew[i, j] = Best_P[1, j] - MOP * ((UB(j) - LB(j)) * Mu + LB(j))
                        else:
                            Xnew[i, j] = Best_P[1, j] + MOP * ((UB(j) - LB(j)) * Mu + LB(j))
            Flag_UB = Xnew[i,:] > UB
            Flag_LB = Xnew[i,:] < LB
            Xnew[i, :] = (np.multiply(Xnew[i,:], (not (Flag_UB + Flag_LB)))) + np.multiply(UB,Flag_UB) + np.multiply(LB, Flag_LB)
            Ffun_new[1, i] = F_obj(Xnew[i,:])
            if Ffun_new[1, i] < Ffun[1, i]:
                X[i, :] = Xnew[i,:]
                Ffun[1, i] = Ffun_new[1, i]
            if Ffun[1, i] < Best_FF:
                Best_FF = Ffun[1, i]
                Best_P = X[i,:]
                # Update the convergence curve
                Conv_curve[C_Iter] = Best_FF
                # Print the best solution details after every 50 iterations
                C_Iter = C_Iter + 1
        ct = time.time()-ct

    return Best_FF, Best_P, Conv_curve,ct