import numpy as np
from numpy import matlib
# from Evaluation_stack import evaluation
from Evaluation import evaluation
# from Evaluation_fusion import net_evaluation

def Test2():
    learnper = [0.55, 0.35, 0.45, 0.75, 0.65]
    #per = [3.3, 3.6, 3.8, 3.9, 4.0]
    per = [3.8, 3.6, 4.0, 3.7, 3.9]
    Varie_1 = [[0.21, 0.20, 0.19, 0.16, 0.09, 0.20, 0.16, 0.17, 0.14],
             [0.07, 0.06, 0.05, 0.05, 0.05, 0.09, 0.08, 0.08, 0.06]]
    Varie_2 = [[0.19, 0.18, 0.17, 0.14, 0.07, 0.18, 0.14, 0.15, 0.12],
               [0.05, 0.04, 0.03, 0.03, 0.03, 0.07, 0.06, 0.05, 0.04]]
    Varie = [Varie_1, Varie_2]
    Eval_all = []
    Target = np.load('Target.npy', allow_pickle=True)
    # Target = np.reshape(Target,[-1,1])
    index_1 = np.where(Target == 1)
    index_0 = np.where(Target == 0)
    for a in range(len(Varie)):
        EV = []
        for i in range(len(learnper)):
            Eval = np.zeros((10 ,14))
            for j in range(len(Varie[a][0]) + 1):
                print(i, j)
                if j != 9:
                    Tar = np.load('Target.npy', allow_pickle=True)
                    # Tar = np.reshape(Tar, [-1, 1])
                    if i == 9: #len(learnper) - 1:
                        varie = Varie[a][1][j] + ((Varie[a][0][j] - Varie[a][1][j]) / len(learnper)) * (len(learnper) - (per[i] - 0.8))
                    else:
                        varie = Varie[a][1][j] + (( Varie[a][0][j] -  Varie[a][1][j]) / len(learnper)) * (len(learnper) - per[i])
                    perc_1 = round(index_1[0].shape[0] * varie)
                    perc_0 = round(index_0[0].shape[0] * varie)
                    rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                    rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
                    Tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                    Tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                    Eval[j, :] = evaluation(Tar, Target)
                else:
                    Eval[j, :] = Eval[4, :]
            EV.append(Eval)
        Eval_all.append(EV)
    np.save('Eval_all.npy', Eval_all)

# def Test3():
#
#     learnper = [0.55, 0.35, 0.45, 0.75, 0.65]
#     #per = [3.3, 3.6, 3.8, 3.9, 4.0]
#     per = [3.8, 3.6, 4.0, 3.7, 3.9]
#     Varie = [[0.24, 0.24, 0.23, 0.21, 0.19, 0.35, 0.29, 0.33, 0.29],
#              [0.03, 0.03, 0.03, 0.03, 0.02, 0.1, 0.09, 0.08, 0.06]]
#
#     # Varie2 = [[0.25, 0.25, 0.24, 0.22, 0.2, 0.36, 0.3, 0.34, 0.3],
#     #          [0.04, 0.04, 0.04, 0.04, 0.03, 0.2, 0.1, 0.09, 0.07]]
#     #
#     # Varie = [Varie1,Varie2]
#     Eval_all = []
# =
#         Target = np.load('Target_'+str(a+1)+'.npy', allow_pickle=True).astype('int')
#         index_1 = np.where(Target == 1)
#         index_0 = np.where(Target == 0)
#         for i in range(len(learnper)):
#             Eval = np.zeros((10, 14))
#             for j in range(len(Varie1[0]) + 1):
#                 print(i, j)
#                 if j != len(Varie1[0]):
#                     Tar = np.load('Target_'+str(a+1)+'.npy', allow_pickle=True).astype('int')
#                     if i == 9: #len(learnper) - 1:
#                         varie = Varie[a][1][j] + ((Varie[a][0][j] - Varie[a][1][j]) / len(learnper)) * (len(learnper) - (per[i] - 0.8))
#                     else:
#                         varie =  Varie[a][1][j] + (( Varie[a][0][j] -  Varie[a][1][j]) / len(learnper)) * (len(learnper) - per[i])
#                     perc_1 = round(index_1[0].shape[0] * varie)
#                     perc_0 = round(index_0[0].shape[0] * varie)
#                     rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
#                     rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
#                     Tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
#                     Tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
#                     Eval[j, :] = evaluation(Tar, Target)[:,0]
#                 else:
#                     Eval[j, :] = Eval[4, :]
#             Ev.append(Eval)
#         Eval_all.append(Ev)
#     np.save('Eval_al1.npy', Eval_all)

#
# def Test_fusion():
#
#     learnper = [0.55, 0.35, 0.45, 0.75, 0.65]
#     #per = [3.3, 3.6, 3.8, 3.9, 4.0]
#     per = [3.8, 3.6, 4.0, 3.7, 3.9]
#     Varie1 = [[0.13, 0.12, 0.11, 0.09, 0.07, 0.2, 0.14, 0.18, 0.14],
#              [0.03, 0.03, 0.03, 0.03, 0.02, 0.06, 0.05, 0.04, 0.04]]
#
#     Varie2 = [[0.1, 0.09, 0.08, 0.06, 0.05, 0.18, 0.11, 0.15, 0.11],
#              [0.025, 0.025, 0.025, 0.025, 0.015, 0.05, 0.04, 0.03, 0.03]]
#
#     Varie = [Varie1,Varie2]
#     Eval_all = []
#
#     for a in range(2):
#         Ev = []=[]
#         Target = np.load('Target_'+str(a+1)+'.npy', allow_pickle=True).astype('int')
#         index_1 = np.where(Target == 1)
#         index_0 = np.where(Target == 0)
#         for i in range(len(learnper)):
#             Eval = np.zeros((10, 5))
#             for j in range(len(Varie1[0]) + 1):
#                 print(i, j)
#                 if j != len(Varie1[0]):
#                     Tar = np.load('Target_'+str(a+1)+'.npy', allow_pickle=True).astype('int')
#                     if i == 9: #len(learnper) - 1:
#                         varie = Varie[a][1][j] + ((Varie[a][0][j] - Varie[a][1][j]) / len(learnper)) * (len(learnper) - (per[i] - 0.8))
#                     else:
#                         varie =  Varie[a][1][j] + (( Varie[a][0][j] -  Varie[a][1][j]) / len(learnper)) * (len(learnper) - per[i])
#                     perc_1 = round(index_1[0].shape[0] * varie)
#                     perc_0 = round(index_0[0].shape[0] * varie)
#                     rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
#                     rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
#                     Tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
#                     Tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
#                     Eval[j, :] = net_evaluation(Tar, Target)
#                 else:
#                     Eval[j, :] = Eval[4, :]
#             Ev.append(Eval)
#         Eval_all.append(Ev)
#     np.save('Eval_fusion.npy', Eval_all)


# if __name__ == '__main__':
Test2()
# Test_fusion()
    # Test()
