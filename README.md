# object_centric_VAD
An Tensorflow Implement of CVPR 2019 "Object-centric Auto-Encoders and Dummy Anomalies for Abnormal Event Detection in Video"
[Paper Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf)

Recently, I haven't get the results as well as the paper mentioned.

## Requirements
> tensorflow >=1.5.0 ( I use tensorflow 1.10.0 )
>
> scikit-learn

## Difference between the Author' Work
- Replace VlFeat with scikit-learn, which is much more popular.

## Framework Overview
The framework include Three Parts:
 1. Object-Detection, use the models release on [object_detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), use Resnet50_fpn_coco, you should download the pretrained model first. 
 2. Three Conv-Auto-Encoder to extract the features of the cropped content, (the detailed design of which can be found in the paper.)
 3. Kmeans Clustering, then train K one-verse-rest Linear SVMs.
 4. Use K OVC SVMs to calculate the anomaly score.
 
 ## Datasets
 You can get the download link from StevenLiu/ano_pred_cvpr2018
 
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
 
 2. train the CAEs
 
 3. clustering and train the SVMs
 
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
