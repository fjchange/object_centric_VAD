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
import pickle
import time
from utils import evaluate

# prefix = '/data/jiachang/'
# if not os.path.exists(prefix):
#     prefix = '/data0/jiachang/'
#     if not os.path.exists(prefix):
#         prefix='/home/manning/'

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default='0', help='Use which gpu?')
    parser.add_argument('-d', '--dataset', type=str, help='Train on which dataset')
    parser.add_argument('-b','--bn',type=bool,default=False,help='whether to use BN layer')
    parser.add_argument('--model_path',type=str,help='Path to saved tensorflow CAE model')
    parser.add_argument('--graph_path',type=str,help='Path to saved object detection frozen graph model')
    parser.add_argument('--svm_model',type=str,help='Path to saved svm model')
    parser.add_argument('--dataset_folder',type=str,help='Dataset Fodlder Path')
    parser.add_argument('-c','--class_add',type=bool,default=False,help='Whether to add class one-hot embedding to the featrue')
    parser.add_argument('-n','--norm',type=int,default=0,help='Whether to use Normalization to the Feature and the normalization level')
    args = parser.parse_args()
    return args


def test(CAE_model_path, OVR_SVM_path, args,gap=2, score_threshold=0.4):
    # to get the image paths
    image_folder =args.dataset_folder
    vids_paths = util.get_vids_paths(image_folder)
    # to set gpu visible
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import tensorflow as tf
    # to load the ssd fpn model, and get related tensor
    object_detection_graph = inference.load_frozen_graph(args.graph_path)
    with object_detection_graph.as_default():
        ops = object_detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = object_detection_graph.get_tensor_by_name(tensor_name)

        image_tensor = object_detection_graph.get_tensor_by_name('image_tensor:0')

        former_batch = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 1], name='former_batch')
        gray_batch = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 1], name='gray_batch')
        back_batch = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 1], name='back_batch')

        grad1_x, grad1_y = tf.image.image_gradients(former_batch)
        # grad2_x,grad2_y=tf.image.image_gradients(gray_batch)
        grad3_x, grad3_y = tf.image.image_gradients(back_batch)

        grad_dis_1 = tf.sqrt(tf.square(grad1_x) + tf.square(grad1_y))
        grad_dis_2 = tf.sqrt(tf.square(grad3_x) + tf.square(grad3_y))

        former_feat =CAE_encoder(grad_dis_1, 'former', bn=args.bn, training=False)
        gray_feat = CAE_encoder(gray_batch, 'gray', bn=args.bn, training=False)
        back_feat = CAE_encoder(grad_dis_2, 'back', bn=args.bn, training=False)
        # [batch_size,3072]
        feat = tf.concat([tf.layers.flatten(former_feat), tf.layers.flatten(gray_feat), tf.layers.flatten(back_feat)],
                         axis=1)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='former_encoder')
        var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gray_encoder'))
        var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='back_encoder'))

        g_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='former_encoder')
        g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gray_encoder'))
        g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='back_encoder'))
        bn_list=[g for g in g_list if 'moving_mean' in g.name or 'moving_variance' in g.name]
        var_list+=bn_list
        restorer = tf.train.Saver(var_list=var_list)

        (image_height, image_width) = util.image_size_map[args.dataset]
        #image_height,image_width=640,640
        clf = joblib.load(OVR_SVM_path)

        anomaly_scores_records = []

        timestamp = time.time()
        num_videos = len(vids_paths)
        total = 0

        with tf.Session() as sess:
            if args.bn:
                restorer.restore(sess, CAE_model_path + '_bn')
            else:
                restorer.restore(sess, CAE_model_path)

            for frame_paths in vids_paths:
                anomaly_scores = np.empty(shape=(len(frame_paths),), dtype=np.float32)

                for frame_iter in range(gap, len(frame_paths)-gap):
                    img = np.expand_dims(util.data_preprocessing(frame_paths[frame_iter],target_size=640), axis=0)
                    output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: img})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.int8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]

                    _temp_anomaly_scores = []
                    _temp_anomaly_score = 0.

                    for score, box in zip(output_dict['detection_scores'], output_dict['detection_boxes']):
                        if score >= score_threshold:
                            box = [int(box[0] * image_height), int(box[1] * image_height), int(box[2] * image_height),
                                   int(box[3] * image_width)]
                            img_gray = util.box_image_crop(frame_paths[frame_iter], box)
                            img_former = util.box_image_crop(frame_paths[frame_iter - gap], box)
                            img_back = util.box_image_crop(frame_paths[frame_iter + gap], box)

                            _feat = sess.run(feat, feed_dict={former_batch: np.expand_dims(img_former, 0),
                                                              gray_batch: np.expand_dims(img_gray, 0),
                                                              back_batch: np.expand_dims(img_back, 0)})
                            if args.norm!=0:
                                _feat=util.norm_(_feat,l=args.norm)
                            if args.class_add:
                                _temp=np.zeros(90,dtype=np.float32)
                                _temp[output_dict['detection_classes'][0][0]-1]=1
                                _feat[0]=np.concatenate((_feat[0],_temp),axis=0)
                            scores = clf.decision_function(_feat)
                            _temp_anomaly_scores.append(-max(scores[0]))
                    if _temp_anomaly_scores.__len__() != 0:
                        _temp_anomaly_score = max(_temp_anomaly_scores)

                    print('video = {} / {}, i = {} / {}, score = {:.6f}'.format(
                        frame_paths[0].split('/')[-2], num_videos, frame_iter, len(frame_paths), _temp_anomaly_score))

                    anomaly_scores[frame_iter] = _temp_anomaly_score

                anomaly_scores[:gap] = anomaly_scores[gap]
                anomaly_scores[-gap:] = anomaly_scores[-gap-1]

#                 min_score=min(anomaly_scores)
#                 for i,_s in enumerate(anomaly_scores):
#                     if _s==100.:
#                         anomaly_scores[i]=min_score
                anomaly_scores_records.append(anomaly_scores)
                total += len(frame_paths)

    # use the evaluation functions from github.com/StevenLiuWen/ano_pred_cvpr2018
    result_dict = {'dataset': args.dataset, 'psnr': anomaly_scores_records, 'flow': [], 'names': [],
                   'diff_mask': []}
    used_time = time.time() - timestamp

    print('total time = {}, fps = {}'.format(used_time, total / used_time))

    # TODO specify what's the actual name of ckpt.
    if not args.bn:
        pickle_path = '/home/'+args.machine+'/anomaly_scores/' + args.dataset + '.pkl'
    else:
        pickle_path = '/home/'+args.machine+'/anomaly_scores/' + args.dataset +'_bn'+ '.pkl'

    with open(pickle_path, 'wb') as writer:
        pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

    results = evaluate.evaluate_all( pickle_path)
    print(results)

if __name__=='__main__':
    args=arg_parse()
    test(args.model_path,args.svm_path,args,score_threshold=0.4)
