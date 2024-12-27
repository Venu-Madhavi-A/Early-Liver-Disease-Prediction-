import os
import nibabel as nib
import numpy as np
from cv2 import resize
from numpy import matlib
import random as rn
from AGTO import AGTO
from AOA import AOA
from A_LSA import A_LSA_segmentation
from CSO import CSO
from EOSA import EOSA
from Global_Vars import Global_Vars
from Image_Results import Image_Results
from Model_Densenet import Model_DENSENET
from Model_MOBILENET import Model_MOBILENET
from Model_RESNET import Model_RESNET
from Model_ViT_DRDNet import Model_ViT_DRDNet
from Objective_Function import Objfun_Cls
from PROPOSED import PROPOSED
from Plot_Results import Plot_ROC, Plot_Results, Confusion_matrix, plot_results_conv
from plot_Segment import Plot_seg_Results



no_of_datasets = 2

#Read Dataset_1
an = 0
if an == 1:
    Target = []
    Images = []
    path = './Datasets/Dataset_1'
    dir_in = os.listdir(path)
    for i in range(len(dir_in)-1):
        print(dir_in[i+1])
        out = path + '/' + dir_in[i+1]
        dir_out = os.listdir(out)
        for j in range(len(dir_out)):
            print(i,j)
            images_2 = out + '/' + dir_out[j]
            img = nib.load(images_2)
            image_data = img.get_fdata()
            original_shape = image_data.shape
            img_count = original_shape[-1]
            new_shape = (512, 512)
            data = resized_img_data = resize(image_data, new_shape, anti_aliasing=True)
            for k in range(len(data)):
                print(k)
                slice_data = data[:, :, i].astype('uint8')
                Images.append(slice_data)
                Target.append(dir_in[i+1])
    Targ = np.asarray(Target)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    np.save('Images_1.npy', Images)
    np.save('Targets_1.npy', tar)


#Read Dataset_2
an = 0
if an == 1:
    dir_in = './Datasets/Dataset_2/'
    direct = os.listdir(dir_in)
    images = []
    target = []
    for i in range(len(direct)):
        file = dir_in + direct[i] + '/'
        if 'imagesTr' in file:
            direct1 = os.listdir(file)
            data1 = []
            for j in range(len(direct1)):
                file1 = file + direct1[j]
                if '._' not in direct1[j]:
                    data1.append(direct1[j])
                else:
                    pass
            for j in range(len(data1), len(data1) + 4):
                file1 = file + direct1[j]
                read = nifti_img = nib.load(file1)
                for k in range(read.shape[2]):
                    image = nifti_img.get_fdata()[:, :, k].astype('uint8')
                    new_shape = (512, 512)
                    images.append(image)
        elif 'labelsTr' in file:
            direct1 = os.listdir(file)
            data1 = []
            for j in range(len(direct1)):
                file1 = file + direct1[j]
                if '._' in direct1[j]:
                    data1.append(direct1[j])
                else:
                    break
            for j in range(len(data1), len(data1) + 4):
                file1 = file + direct1[j]
                read = nifti_img = nib.load(file1)
                for k in range(read.shape[2]):
                    tar = np.where(nifti_img.get_fdata()[:, :, k].astype('uint8') > 0)
                    if len(tar[1]) > 0:
                        targ = 1
                    else:
                        targ = 0
                    target.append(targ)
        else:
            continue
    np.save('Images_2.npy', np.asarray(images))
    np.save('Targets_2.npy', np.asarray(target))


# Optimization for Segmentation
an = 0
if an == 1:
    sol = []
    fitness = []
    for i in range(no_of_datasets):
        Images = np.load('Images_' + str(i+1) + '.npy', allow_pickle=True)
        Target = np.load('Targets_' + str(i+1) + '.npy', allow_pickle=True)
        Global_Vars.Images = Images
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 9
        xmin = matlib.repmat(([-1]), Npop, Chlen)
        xmax = matlib.repmat(([1]), Npop, Chlen)
        # xmin = matlib.repmat(([5, 5, 5, 5]), Npop, 1)
        # xmax = matlib.repmat(([255, 50, 255, 50]), Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
        fname = Objfun_Cls
        max_iter = 50

        print('CSO....')
        [bestfit1, fitness1, bestsol1, Time1] = CSO(initsol, fname, xmin, xmax, max_iter)

        print('AOA....')
        [bestfit2, fitness2, bestsol2, Time2] = AOA(initsol, fname, xmin, xmax, max_iter)

        print('AGTO....')
        [bestfit3, fitness3, bestsol3, Time3] = AGTO(initsol, fname, xmin, xmax, max_iter)

        print('EOSA....')
        [bestfit4, fitness4, bestsol4, Time4] = EOSA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

        sol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        fitness.append([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('Bestsol.npy', sol)
    np.save('Fitness.npy', fitness)

# Segmentation Adaptive Level-set algorithm
an = 0
if an == 1:
    Seg = []
    for i in range(no_of_datasets):
        Images = np.load('Images_' + str(i+1) + '.npy', allow_pickle=True)
        for j in range(len(Images)):
            print('Image', j)
            Segmentation = A_LSA_segmentation(Images[j])
            Seg.append(Segmentation)
        np.save('Segmentation_Images' + str(i + 1) + 'npy', Seg)


## Classification ##
an = 0
if an == 1:
    Eval = []
    for i in range(no_of_datasets):
        Images = np.load('Segmentation_Images' + str(i + 1) + '.npy', allow_pickle=True)
        Target = np.load('Targets_' + str(i + 1) + '.npy', allow_pickle=True)
        sol = np.load('Bestsol.npy',allow_pickle=True)
        activation_function = ['Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid']
        vl = [0, 1, 2, 3, 4]
        for m in range(len(activation_function)):
            per = round(Images.shape[0] * 0.75)
            EVAL = np.zeros((5, 14))
            for i in range(5): # for all algorithms
                train_data = Images[:per, :]
                train_target = Target[:per, :]
                test_data = Images[per:, :]
                test_target = Target[per:, :]
                EVAL[i, :] = Model_ViT_DRDNet(Images, Target, sol[i].astype('int'))
            train_data = Images[:per, :]
            train_target = Target[:per, :]
            test_data = Images[per:, :]
            test_target = Target[per:, :]
            EVAL[0, :] = Model_RESNET(train_data, train_target, test_data, test_target)
            EVAL[1, :] = Model_MOBILENET(train_data, train_target, test_data, test_target)
            EVAL[2, :] = Model_DENSENET(train_data, train_target, test_data, test_target)
            EVAL[3, :] = Model_ViT_DRDNet(Images, Target)
            EVAL[4, :] = EVAL[4, :]
            Eval.append(EVAL)
    np.save('Eval_all.npy', Eval)



Plot_seg_Results()
Plot_ROC()
Plot_Results()
Confusion_matrix()
plot_results_conv()
Image_Results()

