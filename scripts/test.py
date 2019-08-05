import tensorflow as tf
from sklearn import svm
from sklearn.externals import joblib
import sys
sys.path.append('../')
from scripts import inference
from models.CAE import CAE_encoder
from utils import util
import os
import argparse

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
    args=arg_parse()

    image_folder=prefix+args.dataset+'/testing/frames/'
    vids_paths=util.get_vids_paths(image_folder,gap=2)

    for vid in vids_paths:
        for frame_path in vid:


