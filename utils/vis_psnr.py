import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils.evaluate import *

def vis_psnr(loss_file,name=None):


    # assert len(loss_file)==2

        # the name of dataset, loss, and ground truth
    dataset, psnr_records_1, gt = load_psnr_gt(loss_file=loss_file)
    # _,psnr_records_2,_=load_psnr_gt(loss_file=loss_file[1])
    # the number of videos
    num_videos = len(psnr_records_1)

    psnr_max_1=0
    psnr_min_1=1
    # psnr_max_2=0
    # psnr_min_2=1

    for video in psnr_records_1:
        if psnr_max_1<max(video):
            psnr_max_1=max(video)
        if psnr_min_1>min(video):
            psnr_min_1=min(video)

    # for video in psnr_records_2:
    #     if psnr_max_2<max(video):
    #         psnr_max_2=max(video)
    #     if psnr_min_2>min(video):
    #         psnr_min_2=min(video)


    scores_1 = np.array([], dtype=np.float32)
    # scores_2 = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)


    # video normalization
    for i in range(num_videos):
        distance_1 = psnr_records_1[i]
        # distance_2=psnr_records_2[i]
        if NORMALIZE:
            distance_1 -= distance_1.min()  # distances = (distance - min) / (max - min)
            distance_1 /= distance_1.max()
            distance_1 = 1 - distance_1

            # distance_2 -= distance_2.min()  # distances = (distance - min) / (max - min)
            # distance_2 /= distance_2.max()
            # distance_2 = 1 - distance_2
        # if NORMALIZE:
        #     distance_1-=psnr_min_1
        #     distance_1/=(psnr_max_1-psnr_min_1)
        #     distance_2-=psnr_min_2
        #     distance_2/=(psnr_max_2-psnr_min_2)
        scores_1 = np.concatenate((scores_1, distance_1), axis=0)
        # scores_2 = np.concatenate((scores_2, distance_2[DECIDABLE_IDX:]), axis=0)

        labels = np.concatenate((labels, gt[i]), axis=0)

        x=np.arange(distance_1.__len__())

        l1=plt.plot(x,distance_1,'b-',label='origin')
        # l2=plt.plot(x,distance_2[DECIDABLE_IDX:],'g-',label='stack_1')
        # plt.plot(x,distance_2[DECIDABLE_IDX:],'g-')
        plt.plot(x,distance_1,'b-')
        plt.plot(x, gt[i],'r-')
        plt.xlabel('frame_no')
        plt.ylabel('scores')
        plt.title('compare of different type of error calculation of video #{}'.format(i+1))
        plt.legend()
        plt.show(block=True)

    fpr_1, tpr_1, _= metrics.roc_curve(labels, scores_1, pos_label=0)
    # fpr_2,tpr_2,_=metrics.roc_curve(labels,scores_2,pos_label=0)

    l1=plt.plot(fpr_1,tpr_1,'b-',label='origin')
    # l2=plt.plot(fpr_2,tpr_2,'r-',label='recurrence')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('{} ROC Compare'.format(name))
    plt.legend()
    plt.show(block=True)

    # neg_total_distances_1=labels*scores_1
    # pos_total_distances_1=(1-labels)*scores_1
    # neg_total_distances_2=labels*scores_2
    # pos_total_distances_2=(1-labels)*scores_2
    #
    # neg_avg_distance_1=np.average(neg_total_distances_1)
    # pos_avg_distance_1=np.average(pos_total_distances_1)
    # dis_1=pos_avg_distance_1-neg_avg_distance_1
    # neg_avg_distance_2=np.average(neg_total_distances_2)
    # pos_avg_distance_2=np.average(pos_total_distances_2)
    # dis_2=pos_avg_distance_2-neg_avg_distance_2

    # print('neg_avg_distance_1 is {}  pos_avg_distance_1 is {}\n'
    #       'neg_avg_distance_2 is {}  pos_avg_distance_2 is {}'.format(neg_avg_distance_1,pos_avg_distance_1,neg_avg_distance_2,pos_avg_distance_2))
    #print('avg distant between pos and neg of 1 is {}, 2 is {}'.format(dis_1,dis_2))
    # print('avg distant between pos and neg of 1 is {}'.format(dis_1))



if __name__ =='__main__':
    vis_psnr('/home/manning/autor_results/Ionescu_et_al_CVPR_2019_ShanghaiTech_results.pkl',name='avenue')

