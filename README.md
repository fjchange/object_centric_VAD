# object_centric_VAD
An Tensorflow Implement of CVPR 2019 "Object-centric Auto-Encoders and Dummy Anomalies for Abnormal Event Detection in Video"
[Paper Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf)

__Recently, I haven't got the results as awesome as the paper mentioned .__(maybe any porblem in the code, I will be appreciated that if you point it out!)

## Requirements
> tensorflow >=1.5.0 ( I use tensorflow 1.10.0 )
>
> scikit-learn
>
> cyvlfeat (install by anaconda recommended)


## Difference between the Author' Workr 
Considering the author finish the work on Matlab with Vlfeat, there is no complete python version of version available now.
So I
- Replace VlFeat's SVM with Sklearn's  OneVsRestClassifier with SGDClassifier as basic estimizer. ( As the author said, SDCA optimizer and hinge loss work well, but there is no SDCA optimizer in sklearn.)

You can also 
- To use C verision Dynamic Libray of Vlfeat by cffi / cython
- To use tesnorflow to realize the SVM (TF has SDCA optimizer)

## About Score Calculation and Score Smoothing
1. __The method of AUC calculation may leads to unfair comparison__
Author calculate the AUC by calculate all the video's AUC first, and then calculate the mean of them as the AUC of the dataset (which is in utils/evaluate.py compute_auc_averate). The evaluate.py is borrowed from StevenLiuWen/ano_pred_cvpr2018, which concat all the videos first, and then calculate the AUC as the dataset's AUC.

| AUC type | As the author | As Liu et.al |
|----|----|----|
|Avenue| 90.4% | 86.56% |
|ShanghaiTech|84.9%|78.5645%|


2. __Score Smoothing influence the Result Output__
There two parameters in score_smoothing (which is utils/util.py), the parameters can influence the final result.

## Framework Overview
The framework include Three Parts:
 1. Object-Detection, use the models release on [object_detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), use Resnet50_fpn_coco, you should download the pretrained model first. 
 2. Three Conv-Auto-Encoder to extract the features of the cropped content, (the detailed design of which can be found in the paper.)
 3. Kmeans Clustering, then train K one-verse-rest Linear SVMs.
 4. Use K OVC SVMs to calculate the anomaly score.
 
 ## Datasets
 You can get the download link from github.com/StevenLiuWen/ano_pred_cvpr2018
 
 ## Path Setting
 Several paths you need to set as below:
 1. PATH_TO_DATASET_FOLDER
 2. PATH_FOR_CAE_MODEL_TO_SAVE
 3. PATH_TO_OBJECT_DETECTION_MODEL
 4. PATH_FOR_SVM_TO_SAVE
 5. PATH_FOR_SUMMARY_FOR_TENSORBOARD
 
 ## Training
 Training process includes 3 steps:
 1. extract the boxes of the dataset
 > python inference.py --gpu GPU --dataset avenue 
 2. train the CAEs
 > python train.py --gpu GPU --dataset avenue --train CAE
 3. clustering and train the SVMs
  > python train.py --gpu GPU --dataset avenue --train SVM

 ## Testing
 > python test.py --gpu GPU --dataset avenue --model_path YOUR_CAE_MODEL_PATH
 
 ## Reference
 1. Tensorflow Object Detection API
 2. The codes of evaluation part are from github.com/StevenLiuwen/ano_pred_cvpr2018
 3. The project is based on the paper of "Object-centric Auto-encoders and Dummy Anomalies for Abnormal Event Detection in Video"
 
 If you find this useful, please cite works as follows:
```
 misc{object_centrci_VAD,
     author = {Jiachang Feng},
     title = { A Implementation of {Obejct-Centric VAD} Using {Tensorflow}},
     year = {2019},
     howpublished = {\url{https://github.com/fjchange/object_centric_VAD}}
  }
```
