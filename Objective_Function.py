import numpy as np
from Global_Vars import Global_Vars
from Model_ViT_DRDNet import Model_ViT_DRDNet
def Objfun_Cls(Soln):
    images = Global_Vars.Images
    Targ = Global_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        learnper = round(images .shape[0] * 0.75)
        train_data = images[learnper:, :]
        train_target = Targ[learnper:, :]
        test_data = images[:learnper, :]
        test_target = Targ[:learnper, :]
        Eval = Model_ViT_DRDNet(images, Targ, sol.astype('int'))
        Fitn[i] = (1 / Eval[4]) + Eval[9]
    return Fitn


