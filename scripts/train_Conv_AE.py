import sys
sys.path.append('../')

from models import Conv_AE
import argparse
import numpy as np
import tensorflow as tf
from utils.util import Conv_AE_dataset
import os

summary_save_path_pre='/home/jiachang/summary/Conv_AE_'

prefix='/data/jiachang/'
if not os.path.exists(prefix):
    prefix='/data0/jiachang/'
    if not os.path.exists(prefix):
        prefix='/home/manning/'
        summary_save_path_pre = '/home/manning/summary/CAE_'
        svm_save_path_pre = '/home/manning/clfs/'

model_save_path_pre=prefix+'tf_models/Conv_AE_'

batch_size=64
learning_rate=[1e-3,1e-4]
lr_decay_epochs=[100]
epochs=200