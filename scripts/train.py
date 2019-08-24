import numpy as np
import os
import sys
sys.path.append('../')
import argparse
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
import random
from utils import util
import sklearn.svm as svm
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from models import CAE
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from cyvlfeat.kmeans import kmeans,kmeans_quantize


summary_save_path_pre=YOUR_PATH_TO_STORE_SUMMARY#'/home/jiachang/summary/CAE_'
svm_save_dir=YOUR_PATH_TO_STORE_SVM_MODEL#'/home/jiachang/clfs/'

# prefix='/data/jiachang/'
# if not os.path.exists(prefix):
#     prefix='/data0/jiachang/'
#     if not os.path.exists(prefix):
#         prefix='/home/manning/'
#         summary_save_path_pre = '/home/manning/summary/CAE_'
#         svm_save_path_pre = '/home/manning/clfs/'

model_save_path_pre='CAE_'

batch_size=64
learning_rate=[1e-3,1e-4]
lr_decay_epochs=[100]
epochs=200
'''
The Author said that the model may be better when 90-d one-hot embedding, representing the object class in COCO, 
add to the feature vector, which is can be activated by '--class_add'
'''
def arg_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-g','--gpu',type=str,default='0',help='Use which gpu?')
    parser.add_argument('-d','--dataset',type=str,help='Train on which dataset')
    parser.add_argument('-t','--train',type=str,help='Train on SVM / CAE')
    parser.add_argument('-b','--bn',type=bool,default=False,help='whether to use BN layer')
    parser.add_argument('--dataset_folder',type=str,help='Dataset Fodlder Path')
    parser.add_argument('--model_dir',type=str,help='Folder to save tensorflow CAE model')
    parser.add_argument('-c','--class_add',type=bool,default=False,help='Whether to add class one-hot embedding to the featrue')
    parser.add_argument('-n','--norm',type=int,default=0,help='Whether to use Normalization to the Feature and the normalization level')
    parser.add_argument('--box_imgs_npy_path',type=str,help='Path for npy file that store the \(box,img_path\)')
    args=parser.parse_args()
    return args

def train_CAE(path_boxes_np,args):
    epoch_len=len(np.load(path_boxes_np))
    f_imgs,g_imgs,b_imgs=util.CAE_dataset_feed_dict(prefix,path_boxes_np,dataset_name=args.dataset)
    #former_batch,gray_batch,back_batch=util.CAE_dataset(path_boxes_np,args.dataset,epochs,batch_size)
    former_batch=tf.placeholder(dtype=tf.float32,shape=[batch_size,64,64,1],name='former_batch')
    gray_batch=tf.placeholder(dtype=tf.float32,shape=[batch_size,64,64,1],name='gray_batch')
    back_batch=tf.placeholder(dtype=tf.float32,shape=[batch_size,64,64,1],name='back_batch')

    grad1_x,grad1_y=tf.image.image_gradients(former_batch)
    # grad2_x,grad2_y=tf.image.image_gradients(gray_batch)
    grad3_x,grad3_y=tf.image.image_gradients(back_batch)

    grad_dis_1=tf.sqrt(tf.square(grad1_x)+tf.square(grad1_y))
    grad_dis_2=tf.sqrt(tf.square(grad3_x)+tf.square(grad3_y))

    former_outputs=CAE.CAE(grad_dis_1,'former',bn=args.bn,training=True)
    gray_outputs=CAE.CAE(gray_batch,'gray',bn=args.bn,training=True)
    back_outputs=CAE.CAE(grad_dis_2,'back',bn=args.bn,training=True)

    former_loss=CAE.pixel_wise_L2_loss(former_outputs,grad_dis_1)
    gray_loss=CAE.pixel_wise_L2_loss(gray_outputs,gray_batch)
    back_loss=CAE.pixel_wise_L2_loss(back_outputs,grad_dis_2)

    global_step=tf.Variable(0,dtype=tf.int32,trainable=False)
    global_step_a=tf.Variable(0,dtype=tf.int32,trainable=False)
    global_step_b=tf.Variable(0,dtype=tf.int32,trainable=False)

    lr_decay_epochs[0] =int(epoch_len//batch_size)*lr_decay_epochs[0]

    lr=tf.train.piecewise_constant(global_step,boundaries=lr_decay_epochs,values=learning_rate)

    former_vars=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope='former_')
    gray_vars=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope='gray_')
    back_vars=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope='back_')
    # print(former_vars)

    former_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(former_loss,var_list=former_vars,global_step=global_step)
    gray_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(gray_loss,var_list=gray_vars,global_step=global_step_a)
    back_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(back_loss,var_list=back_vars,global_step=global_step_b)

    step=0
    if not args.bn:
        writer=tf.summary.FileWriter(logdir=summary_save_path_pre+args.dataset)
    else:
        writer=tf.summary.FileWriter(logdir=summary_save_path_pre+args.dataset+'_bn')

    tf.summary.scalar('loss/former_loss',former_loss)
    tf.summary.scalar('loss/gray_loss',gray_loss)
    tf.summary.scalar('loss/back_loss',back_loss)
    tf.summary.image('inputs/former',grad_dis_1)
    tf.summary.image('inputs/gray',gray_batch)
    tf.summary.image('inputs/back',grad_dis_2)
    tf.summary.image('outputs/former',former_outputs)
    tf.summary.image('outputs/gray',gray_outputs)
    tf.summary.image('outputs/back',back_outputs)
    summary_op=tf.summary.merge_all()

    saver=tf.train.Saver(var_list=tf.global_variables())
    indices=list(range(epoch_len))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            random.shuffle(indices)
            for i in range(epoch_len//batch_size):
                feed_dict={former_batch:[f_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]],
                           gray_batch:[g_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]],
                           back_batch:[b_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]]
                           }
                step,_lr,_,_,_,_former_loss,_gray_loss,_back_loss=sess.run([global_step,lr,former_op,gray_op,back_op,former_loss,gray_loss,back_loss],feed_dict=feed_dict)
                if step%10==0:
                    print('At step {}'.format(step))
                    print('\tLearning Rate {:.4f}'.format(_lr))
                    print('\tFormer Loss {:.4f}'.format(_former_loss))
                    print('\tGray Loss {:.4f}'.format(_gray_loss))
                    print('\tBack Loss {:.4f}'.format(_back_loss))

                if step%50==0:
                    _summary=sess.run(summary_op,feed_dict=feed_dict)
                    writer.add_summary(_summary,global_step=step)
        if not args.bn:
            saver.save(sess,model_save_path_pre+args.dataset)
        else:
            saver.save(sess,model_save_path_pre+args.dataset+'_bn')

        print('train finished!')
        sess.close()

def extract_features(path_boxes_np,CAE_model_path,args):
    f_imgs,g_imgs,b_imgs=util.CAE_dataset_feed_dict(prefix,path_boxes_np,args.dataset)
    print('dataset loaded!')
    iters=np.load(path_boxes_np).__len__()

    former_batch=tf.placeholder(dtype=tf.float32,shape=[1,64,64,1],name='former_batch')
    gray_batch=tf.placeholder(dtype=tf.float32,shape=[1,64,64,1],name='gray_batch')
    back_batch=tf.placeholder(dtype=tf.float32,shape=[1,64,64,1],name='back_batch')

    grad1_x, grad1_y = tf.image.image_gradients(former_batch)
    # grad2_x,grad2_y=tf.image.image_gradients(gray_batch)
    grad3_x, grad3_y = tf.image.image_gradients(back_batch)

    grad_dis_1 = tf.sqrt(tf.square(grad1_x) + tf.square(grad1_y))
    grad_dis_2 = tf.sqrt(tf.square(grad3_x) + tf.square(grad3_y))

    former_feat=CAE.CAE_encoder(grad_dis_1,'former',bn=args.bn,training=False)
    gray_feat=CAE.CAE_encoder(gray_batch,'gray',bn=args.bn,training=False)
    back_feat=CAE.CAE_encoder(grad_dis_2,'back',bn=args.bn,training=False)
    # [batch_size,3072]
    feat=tf.concat([tf.layers.flatten(former_feat),tf.layers.flatten(gray_feat),tf.layers.flatten(back_feat)],axis=1)

    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='former_encoder')
    var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='gray_encoder'))
    var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='back_encoder'))

    g_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='former_encoder')
    g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gray_encoder'))
    g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='back_encoder'))
    bn_list = [g for g in g_list if 'moving_mean' in g.name or 'moving_variance' in g.name]
    var_list += bn_list

    restorer=tf.train.Saver(var_list=var_list)
    data=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.bn:
            restorer.restore(sess, CAE_model_path+'_bn')
        else:
            restorer.restore(sess,CAE_model_path)
        for i in range(iters):
            feed_dict={former_batch:np.expand_dims(f_imgs[i],0),
                       gray_batch:np.expand_dims(g_imgs[i],0),
                       back_batch:np.expand_dims(b_imgs[i],0)}
            result=sess.run(feat,feed_dict=feed_dict)

            if args.norm==0:
                _temp=result[0]
            else:
                _temp=util.norm_(result[0],l=args.norm)[0]

            if args.class_add:
                c_onehot_embedding=np.zeros(90,dtype=np.float32)
                c_onehot_embedding[class_indexes[i]-1]=1
                _temp=np.concatenate((_temp,c_onehot_embedding),axis=0)
                
            data.append(_temp)
        data=np.array(data)
        sess.close()

    return data

def train_one_vs_rest_SVM(path_boxes_np,CAE_model_path,K,args):
    data=extract_features(path_boxes_np,CAE_model_path,args)
    print('feature extraction finish!')
    # clusters, the data to be clustered by Kmeans
    # clusters=KMeans(n_clusters=K,init='k-means++',n_init=10,algorithm='full',max_iter=300).fit(data)
    centers=kmeans(data,num_centers=K,initialization='PLUSPLUS',num_repetitions=10,
                   max_num_comparisons=100,max_num_iterations=100,algorithm='LLOYD',num_trees=3)
    labels=kmeans_quantize(data,centers)

    # to get the sparse matrix of labels
    sparse_labels=np.eye(K)[labels]
    sparse_labels=(sparse_labels-0.5)*2

    # nums=np.zeros(10,dtype=int)
    # for item in clusters.labels_:
    #     nums[item]+=1
    # print(nums)
    print('clustering finished!')
    # SGDC classifier with onevsrest classifier to replace the ovc-svm with hinge loss and SDCA optimizer in the paper
    base_estimizer=SGDClassifier(max_iter=10000,warm_start=True,loss='hinge',early_stopping=True,n_iter_no_change=50,l1_ratio=0)
    ovr_classifer=OneVsRestClassifier(base_estimizer)

    #clf=svm.LinearSVC(C=1.0,multi_class='ovr',max_iter=len(labels)*5,loss='hinge',)
    ovr_classifer.fit(data,sparse_labels)
    joblib.dump(ovr_classifer,svm_save_dir+args.dataset+'.m')
    print('train finished!')


if __name__=='__main__':
    args=arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    # train CAE first than, train SVM
    if args.train=='CAE':
        train_CAE(args.box_imgs_npy_path,args)
    else:
        train_one_vs_rest_SVM(args.box_imgs_npy_path,os.path.join(args.model_dir,model_save_path_pre+args.dataset),10,args)


