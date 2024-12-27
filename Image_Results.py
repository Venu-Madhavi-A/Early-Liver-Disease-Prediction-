import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


no_of_dataset = 2
def Image_Results():
    for i in range(no_of_dataset):
        Orig = np.load('Images_' + str(i+1) +'.npy',allow_pickle=True)
        Image1 = np.load('VGG16_' + str(i+1) +'.npy', allow_pickle=True)
        Image2 = np.load('GRADIENT_BOOSTING_' + str(i+1) +'.npy', allow_pickle=True)
        Image3 = np.load('LDA_' + str(i+1) +'.npy', allow_pickle=True)
        Image4 = np.load('LSA_' + str(i+1) +'.npy', allow_pickle=True)
        segment = np.load('PROPOSED_' + str(i+1) +'.npy', allow_pickle=True)
        # ind = [159, 218, 226, 230, 273]
        for j in range(5):
            original = Orig[j]
            image1 = Image1[j]
            image2 = Image2[j]
            image3 = Image3[j]
            image4 = Image4[j]
            seg = segment[j]
            Output1 = np.zeros((image1.shape)).astype('uint8')
            ind1 = np.where(image1 > 0)
            Output1[ind1] = 255

            Output2 = np.zeros((image2.shape)).astype('uint8')
            ind2 = np.where(image2 > 0)
            Output2[ind2] = 255

            Output3 = np.zeros((image3.shape)).astype('uint8')
            ind3 = np.where(image3 > 0)
            Output3[ind3] = 255

            Output4 = np.zeros((image4.shape)).astype('uint8')
            ind4 = np.where(image4 > 0)
            Output4[ind4] = 255

            Output5 = np.zeros((seg.shape)).astype('uint8')
            ind5 = np.where(seg > 0)
            Output5[ind5] = 255

            fig, ax = plt.subplots(1, 4)
            plt.suptitle("Image %d" % (j + 1), fontsize=20)
            plt.subplot(2, 3, 1)
            plt.title('Original')
            plt.imshow(original)
            plt.subplot(2, 3, 2)
            plt.title('VGG16')
            plt.imshow(Output1)
            plt.subplot(2, 3, 3)
            plt.title('GRADIENT_BOOSTING')
            plt.imshow(Output2)

            plt.subplot(2, 3, 4)
            plt.title('LDA')
            plt.imshow(Output3)

            plt.subplot(2, 3, 5)
            plt.title('LSA')
            plt.imshow(Output4)

            plt.subplot(2, 3, 6)
            plt.title('PROPOSED')
            plt.imshow(Output5)
            # path1 = "./Results/Images/Dataset_%simage.png" % (j + 1)
            # plt.savefig(path1)
            plt.show()
            cv.imwrite('./Results/Images/Original-'+ str(i+1) + '-' + str(j + 1) + '.png', original)
            cv.imwrite('./Results/Images/VGG16-' + str(i+1) + '-' + str(j + 1) + '.png', Output1)
            cv.imwrite('./Results/Images/GRADIENT_BOOSTING-' + str(i+1) + '-' + str(j + 1) + '.png', Output2)
            cv.imwrite('./Results/Images/LDA-' + str(i+1) + '-' + str(j + 1) + '.png', Output3)
            cv.imwrite('./Results/Images/LSA-' + str(i+1) + '-' + str(j + 1) + '.png', Output4)
            cv.imwrite('./Results/Images/PROPOSED-' + str(i+1) + '-' + str(j + 1) + '.png', Output5)

if __name__ == '__main__':
    Image_Results()