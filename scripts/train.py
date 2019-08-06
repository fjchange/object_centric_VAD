import numpy as np
import os
import sys
sys.path.append('../')
import argparse
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

from utils import util
import sklearn.svm as svm
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from models import CAE

prefix='/data/jiachang/'
if not os.path.exists(prefix):
    prefix='/data0/jiachang/'

model_save_path_pre=prefix+'tf_models/CAE_'
summary_save_path_pre=prefix+'summary/CAE_'

svm_save_path_pre=prefix+'clfs/'

batch_size=64
learning_rate=[1e-3,1e-4]
lr_decay_epochs=[100]
epochs=200

def arg_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-g','--gpu',type=str,default='0',help='Use which gpu?')
    parser.add_argument('-d','--dataset',type=str,help='Train on which dataset')
    args=parser.parse_args()
    return args

def train_CAE(path_boxes_np,args):
    epoch_len=len(np.load(path_boxes_np))
    f_imgs,g_imgs,b_imgs=util.CAE_dataset_feed_dict(path_boxes_np,dataset_name=args.dataset)
    #former_batch,gray_batch,back_batch=util.CAE_dataset(path_boxes_np,args.dataset,epochs,batch_size)
    former_batch=tf.placeholder(dtype=tf.float32,shape=[batch_size,64,64,1],name='former_batch')
    gray_batch=tf.placeholder(dtype=tf.float32,shape=[batch_size,64,64,1],name='gray_batch')
    back_batch=tf.placeholder(dtype=tf.float32,shape=[batch_size,64,64,1],name='back_batch')

    grad1_x,grad1_y=tf.image.image_gradients(former_batch)
    grad2_x,grad2_y=tf.image.image_gradients(gray_batch)
    grad3_x,grad3_y=tf.image.image_gradients(back_batch)

    grad_dis_1=tf.sqrt(tf.square(grad2_x-grad1_x)+tf.square(grad2_y-grad1_y))
    grad_dis_2=tf.sqrt(tf.square(grad3_x-grad2_x)+tf.square(grad3_y-grad2_y))

    former_outputs=CAE.CAE(grad_dis_1,'former')
    gray_outputs=CAE.CAE(gray_batch,'gray')
    back_outputs=CAE.CAE(grad_dis_2,'back')

    former_loss=CAE.pixel_wise_L2_loss(former_outputs,former_batch)
    gray_loss=CAE.pixel_wise_L2_loss(gray_outputs,gray_batch)
    back_loss=CAE.pixel_wise_L2_loss(back_outputs,back_batch)

    global_step=tf.Variable(0,dtype=tf.int32,trainable=False)
    lr_decay_epochs[0] =int(epoch_len//batch_size)*lr_decay_epochs[0]

    lr=tf.train.piecewise_constant(global_step,boundaries=lr_decay_epochs,values=learning_rate)

    former_vars=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope='former_')
    gray_vars=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope='gray_')
    back_vars=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope='back_')
    # print(former_vars)

    former_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(former_loss,var_list=former_vars)
    gray_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(gray_loss,var_list=gray_vars)
    back_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(back_loss,var_list=back_vars)

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

    saver=tf.train.Saver(var_list=tf.global_variables())
    indices=np.arange(start=0,stop=epoch_len,step=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            np.random.shuffle(indices)
            for i in range(epoch_len//batch_size):
                feed_dict={former_batch:f_imgs[indices[i*batch_size:(i+1)*batch_size]],
                           gray_batch:g_imgs[indices[i*batch_size:(i+1)*batch_size]],
                           back_batch:b_imgs[indices[i*batch_size:(i+1)*batch_size]]
                           }
                step,_,_,_,_former_loss,_gray_loss,_back_loss=sess.run([global_step,former_op,gray_op,back_op,former_loss,gray_loss,back_loss],feed_dict=feed_dict)
                if step%10==0:
                    print('At step {}'.format(step))
                    print('\tFormer Loss {.4f}'.format(_former_loss))
                    print('\tGray Loss {.4f}'.format(_gray_loss))
                    print('\tBack Loss {.4f}'.format(_back_loss))

                if step%50==0:
                    _summary=sess.run(summary_op,feed_dict=feed_dict)
                    writer.add_summary(_summary,global_step=step)

        saver.save(sess,model_save_path_pre+args.dataset,global_step=step)

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

def extract_features(path_boxes_np,CAE_former_path,CAE_gray_path,CAE_back_path,args):
    former_batch,gray_batch,back_batch=util.CAE_dataset(path_boxes_np,args.dataset,1,1)
    iters=np.load(path_boxes_np).__len__()
    former_feat=CAE.CAE_encoder(former_batch,'former')
    gray_feat=CAE.CAE_encoder(gray_batch,'gray')
    back_feat=CAE.CAE_encoder(back_batch,'back')

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

def train_one_vs_rest_SVM(path_boxes_np,CAE_former_path,CAE_gray_path,CAE_back_path,K,args):
    data=extract_features(path_boxes_np,CAE_former_path,CAE_gray_path,CAE_back_path)
    # clusters, the data to be clustered by Kmeans
    clusters=KMeans(n_clusters=K,init='k-means++',n_init=10,algorithm='full').fit(data)

    # One-Verse-Rest SVM: to train OVC-SVM for
    clfs=[]
    for i in range(K):
        clfs.append(svm.LinearSVC(C=1.0))
        Y=_get_Y(clusters.labels_,i)
        clfs[i].fit(data,Y)

        joblib.dump(clfs[i],svm_save_path_pre+args.dataset+'_'+str(i)+'.m')


if __name__=='__main__':
    args=arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    train_CAE('/home/jiachang/'+args.dataset+'_img_path_box.npy',args)



