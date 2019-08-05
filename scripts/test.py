from sklearn import svm
from sklearn.externals import joblib
import sys
sys.path.append('../')
from scripts import inference
from models.CAE import CAE_encoder
from utils import util
import os
import argparse
import numpy as np

prefix='/data/jiachang/'
if not os.path.exists(prefix):
    prefix='/data0/jiachang/'

def arg_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-g','--gpu',type=str,default='0',help='Use which gpu?')
    parser.add_argument('-d','--dataset',type=str,help='Train on which dataset')
    args=parser.parse_args()
    return args

def test(CAE_path_list,OVR_SVM_path_list):
    # to get the argumentations
    args=arg_parse()
    # to get the image paths
    image_folder=prefix+args.dataset+'/testing/frames/'
    vids_paths=util.get_vids_paths(image_folder,gap=2)
    # to set gpu visible
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import tensorflow as tf
    # to load the ssd fpn model, and get related tensor
    graph=inference.load_frozen_graph()

    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


    for frame_paths in vids_paths:
        for frame_path in frame_paths:
            img=np.expand_dims(util.data_preprocessing(frame_path),axis=0)

            with tf.Session() as sess:



