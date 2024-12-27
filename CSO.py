import time
import numpy as np


def CSO(partMat, fname, LB, UB, itMax):
    # CSO parammeters
    popSz, dimns = partMat.shape

    MR = 0.75

    # here it is ratio of total seeking mode cats to total popSz
    CDC = 0.65

    SRD = 0.25

    SMP = 5
    c = 2.05
    wMax = 0.9
    wMin = 0.3
    w = np.linspace(wMax, wMin, itMax)
    # Random initialization of population between lower and upper bound. Here
    # PartMat is of size population x dimensions
    # initialization of velocity between lower and upper bound based on
    # particle matrix and percentage as specified
    perCnt = 0.25
    vLb = LB[1,:]
    vUp = UB[1,:]
    # velMat = (vUp - vLb).*rand(popSz, dimns) + vLb;
    # Evaluate first g_best based on only intialisation of random poplulation
    fun = fname( partMat)
    f0=[]
    for i in range(fun.shape[0]):
        f0.append(fun[i])
    trackMode=[]
    seekMode=[]
    f0 = np.asarray(f0)
    # f0 = np.reshape(f0,[len(f0),1])
    index0 = np.amin(f0)
    g_best = index0
    oldMinFitval = index0
    for iter in range(itMax):
        # Seeking mode ============================================================
        numOf_skmCats = np.round(popSz * MR)
        # skmCats = randi([1 popSz], numOf_skmCats, 1); # decide seeking mode cats
        skmCats =np.random.randint(numOf_skmCats.astype(int),popSz)
        skmCats_Mat = partMat[skmCats,:]
        skmCatsUp = skmCats_Mat
        # =========================================================================
        # Tracking mode ===========================================================
        trkmCats = np.arange(popSz)
        # trkmCats[skmCats] = []
        trkmCats_Mat = partMat[trkmCats,:]
        if iter == 1:
            trkmCats_vMat = perCnt * trkmCats_Mat
        trkmCats_MatUp = trkmCats_Mat
        # =========================================================================
        # Combining both seeking mode and tracking mode cats ======================
        partMatUp = skmCatsUp
        trkmCats_MatUp[trkmCats, :] = trkmCats_MatUp
        # boundary condition check
        partMatUp_bChk = partMat
        fun2 = fname( partMatUp_bChk)
        # =========================================================================
        # Evaluating new fitness values ===========================================
        fup = []
        for i in range(fun2.shape[0]):
            fup.append(fun2[i])
        newMinFitval = np.amin(fup)
        newMinInx = np.where(np.amin(fup) == fup)
        if newMinFitval < oldMinFitval:
            g_best = partMatUp_bChk[newMinInx,:]
            oldMinFitval = newMinFitval
            partMat = partMatUp_bChk
            velMat = perCnt * partMat
            trkmCats_vMat = trkmCats_MatUp
            bestfit = newMinFitval
        else:
            bestfit = oldMinFitval
            # do nothing
        fitnessVal = bestfit
        best_sols= g_best[1,:]

        ct = time.time()
        return bestfit, fitnessVal, best_sols, ct

        # Boundary constraints

def setBounds(s, Lb, Ub):
    # Apply the lower bound
    ns_tmp = s
    I = ns_tmp < Lb
    ns_tmp[I] = Lb(I)
    # Apply the upper bounds
    J = ns_tmp > Ub
    ns_tmp[J] = Ub(J)
    # Update this new move
    s = ns_tmp
    return s

