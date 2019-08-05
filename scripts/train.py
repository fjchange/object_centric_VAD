from models import CAE
import numpy as np
import os
import os.path as osp
from utils import util
import sklearn.svm as svm
from sklearn.cluster import KMeans
import argparse
from sklearn.externals import joblib
from models import CAE

prefix='/data/jiachang/'
if not os.path.exists(prefix):
    prefix='/data0/jiachang/'

model_save_path_pre=prefix+'tf_models/CAE_'
summary_save_path_pre=prefix+'summary/CAE_'
dataset='avenue'
data_path=prefix+dataset+'/training/frames/'

svm_save_path_pre=prefix+'clfs/'

batch_size=64
learning_rate=[1e-3,1e-4]
lr_decay_epochs=[100]

def arg_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-g','--gpu',type=str,default='0',help='Use which gpu?')
    parser.add_argument('-d','--dataset',type=str,help='Train on which dataset')
    parser.add_argument('-e','--epochs',type=int,help='Train how many times?')
    args=parser.parse_args()
    return args

def train_CAE(path_boxes_np):
    args=arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    epoch_len=len(np.load(path_boxes_np))
    import tensorflow as tf
    former_batch,gray_batch,back_batch=util.CAE_dataset(path_boxes_np,args.dataset,args.epochs,batch_size)

    former_outputs=CAE.CAE(former_batch,'former')
    gray_outputs=CAE.CAE(gray_batch,'gray')
    back_outputs=CAE.CAE(back_batch,'back')

    former_loss=CAE.pixel_wise_L2_loss(former_outputs,former_batch)
    gray_loss=CAE.pixel_wise_L2_loss(gray_outputs,gray_batch)
    back_loss=CAE.pixel_wise_L2_loss(back_outputs,back_batch)

    global_step=tf.Variable(0,trainable=False)
    lr=tf.train.piecewise_constant_decay(global_step,lr_decay_epochs*int(epoch_len//batch_size),learning_rate)

    former_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(former_loss)
    gray_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(gray_loss)
    back_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(back_loss)

    step=0
    writer=tf.summary.FileWriter(logdir=summary_save_path_pre+args.dataset)
    tf.summary.scalar('loss/former_loss',former_loss)
    tf.summary.scalar('loss/gray_loss',gray_loss)
    tf.summary.scalar('loss/back_loss',back_loss)
    tf.summary.image('inputs/former',former_batch)
    tf.summary.image('inputs/gray',gray_batch)
    tf.summary.image('inputs/back',back_batch)
    tf.summary.image('outputs/former',former_outputs)
    tf.summary.image('outputs/gray',gray_outputs)
    tf.summary.image('outputs/back',back_outputs)
    summary_op=tf.summary.merge_all()

    slim=tf.contrib.slim

    former_saver=tf.train.Saver(var_list=slim.get_variables_to_restore(include=('former_encoder','former_decoder')))
    gray_saver=tf.train.Saver(var_list=slim.get_variables_to_restore(include=('gary_encoder','gray_decoder')))
    back_saver=tf.train.Saver(var_list=slim.get_variables_to_restore(include=('back_encoder','back_decoder')))

    with tf.Session() as sess:
        while step<args.epochs*(epoch_len//batch_size):
            step,_,_,_,_former_loss,_gray_loss,_back_loss=sess.run([global_step,former_op,gray_op,back_op,former_loss,gray_loss,back_loss])
            if step%10==0:
                print('At step {}'.format(step))
                print('\tFormer Loss {.4f}'.format(_former_loss))
                print('\tGray Loss {.4f}'.format(_gray_loss))
                print('\tBack Loss {.4f}'.format(_back_loss))

            if step%50==0:
                _summary=sess.run(summary_op)
                writer.add_summary(_summary,global_step=step)

        former_saver.save(sess,model_save_path_pre+'former.ckpt',global_step=step)
        gray_saver.save(sess,model_save_path_pre+'gray.ckpt',global_step=step)
        back_saver.save(sess,model_save_path_pre+'back.ckpt',global_step=step)

        print('train finished!')
        sess.close()

def _get_Y(labels,k):
    Y=labels
    for i in range(labels.shape[0]):
        if labels[i]==k:
            Y[i]=-1
        else:
            Y[i]=1
    return Y

def extract_features(path_boxes_np,CAE_former_path,CAE_gray_path,CAE_back_path):
    args=arg_parse()
    former_batch,gray_batch,back_batch=util.CAE_dataset(path_boxes_np,args.dataset,1,1)
    iters=np.load(path_boxes_np).__len__()
    former_feat=CAE.CAE_encoder(former_batch,'former')
    gray_feat=CAE.CAE_encoder(gray_batch,'gray')
    back_feat=CAE.CAE_encoder(back_batch,'back')

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import tensorflow as tf

    feat=tf.concat([tf.layers.flatten(former_feat),tf.layers.flatten(gray_feat),tf.layers.flatten(back_feat)],axis=1)

    slim=tf.contrib.slim

    former_restorer=tf.train.Saver(var_list=slim.get_variables_to_restore(include='former_encoder'))
    gray_restorer=tf.train.Saver(var_list=slim.get_variables_to_restore(include='gray_encoder'))
    back_restorer=tf.train.Saver(var_list=slim.get_variables_to_restore(include='back_encoder'))

    data=[]
    with tf.Session() as sess:
        former_restorer.restore(sess,CAE_former_path)
        gray_restorer.restore(sess,CAE_gray_path)
        back_restorer.restore(sess,CAE_back_path)
        for i in range(iters):
            data.append(sess.run(feat))
        data=np.array(data)
        sess.close()

    return data

def train_one_vs_rest_SVM(path_boxes_np,CAE_former_path,CAE_gray_path,CAE_back_path,K):
    data=extract_features(path_boxes_np,CAE_former_path,CAE_gray_path,CAE_back_path)
    # clusters, the data to be clustered by Kmeans
    clusters=KMeans(n_clusters=K,init='k-means++',n_init=10,algorithm='full').fit(data)

    # One-Verse-Rest SVM: to train OVC-SVM for
    clfs=[]
    for i in range(K):
        clfs.append(svm.LinearSVC(C=1.0))
        Y=_get_Y(clusters.labels_,i)
        clfs[i].fit(data,Y)

        joblib.dump(clfs[i],svm_save_path_pre+dataset+'_'+str(i)+'.m')


if __name__=='__name__':
    train_CAE('')



