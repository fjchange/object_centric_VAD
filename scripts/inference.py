import numpy as np
import os
import sys
sys.path.append('../')
import tensorflow as tf
from utils import util
from object_detection_utils import label_map_util
from object_detection_utils import visualization_utils as vis_util
import cv2
import argparse

MODEL_NAME='/home/jiachang/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'

PATH_TO_FROZEN_GRAPH=MODEL_NAME+'/frozen_inference_graph.pb'
PATH_TO_LABELS='../utils/mscoco_label_map.pbtxt'

prefix='/data/jiachang/'
if not os.path.exists(prefix):
    prefix='/data0/jiachang/'
    if not os.path.exists(prefix):
        prefix='/hdd/fjc/'

def arg_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-g','--gpu',type=str,default='0',help='Use which gpu?')
    parser.add_argument('-d','--dataset',type=str,help='Train on which dataset')
    args=parser.parse_args()
    return args

def load_frozen_graph():
    detection_graph=tf.Graph()
    with detection_graph.as_default():
        od_graph_def=tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH,'rb')as fid:
            serialized_graph=fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')
    return detection_graph

def run_inference_for_images_per_image(graph,image_folder,np_boxes_path,score_threshold):
    frame_lists=util.get_frames_paths(image_folder,gap=2)
    with graph.as_default():
        ops=tf.get_default_graph().get_operations()
        all_tensor_names={output.name for op in ops for output in op.outputs}
        tensor_dict={}
        for key in [
            'num_detections','detection_boxes','detection_scores',
            'detection_classes'
        ]:
            tensor_name=key+':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key]=tf.get_default_graph().get_tensor_by_name(tensor_name)

        image_tensor=tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        with tf.Session() as sess:
            path_box_lists=[]
            count=0
            for i,frame_path in enumerate(frame_lists):
                image=util.data_preprocessing(frame_path)
                image=np.expand_dims(image,axis=0)

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                # output_dict['detection_classes'] = output_dict[
                #     'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                for score,box in zip(output_dict['detection_scores'],output_dict['detection_boxes']):
                    if score>=score_threshold:
                        path_box_lists.append([frame_path,box])
                        count+=1
                        print('{} added!'.format(count))
            sess.close()
            np.save(np_boxes_path,path_box_lists)
            print('finish boxes detection!')

def vis_detection_result(graph,image_path,output_image_path):
    with graph.as_default():
        ops=tf.get_default_graph().get_operations()
        all_tensor_names={output.name for op in ops for output in op.outputs}
        tensor_dict={}
        for key in [
            'num_detections','detection_boxes','detection_scores',
            'detection_classes','detection_masks'
        ]:
            tensor_name=key+':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key]=tf.get_default_graph().get_tensor_by_name(tensor_name)

        image_tensor=tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        with tf.Session() as sess:
            print('get in the session')
            image = util.data_preprocessing(image_path)
            image_np = np.expand_dims(image, axis=0)
            output_dict=sess.run(tensor_dict,feed_dict={image_tensor:image_np})
            print(output_dict)
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            print(output_dict)
            return output_dict

            # category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
            #
            # image=vis_util.visualize_boxes_and_labels_on_image_array(
            #     image,
            #     output_dict['detection_boxes'],
            #     output_dict['detection_classes'],
            #     output_dict['detection_scores'],
            #     category_index,
            #     instance_masks=output_dict.get('detection_masks'),
            #     use_normalized_coordinates=True,
            #     line_thickness=8)
            #
            # cv2.imwrite(output_image_path,image)

            #sess.close()


def run_inference_get_feature(graph,image_folder):
    frame_lists=util.get_frames_paths(image_folder,gap=2)

    with graph.as_default():
        ops=tf.get_default_graph().get_operations()
        all_tensor_names={output.name for op in ops for output in op.outputs}
        tensor_dict={}
        for key in [
            'num_detections','detection_boxes','detection_scores',
            'detection_classes','detection_masks'
        ]:
            tensor_name=key+':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key]=tf.get_default_graph().get_tensor_by_name(tensor_name)

        image_tensor=tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    #     with tf.Session() as sess:
    #
    # pass

if __name__=='__main__':
    args=arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    # np_paths_boxes_path = prefix+args.dataset+'_'+'img_path_box.npy'
    image_dataset_path=prefix+args.dataset+'/training/frames/'
    print(image_dataset_path)
    graph=load_frozen_graph()
    frame_lists=util.get_frames_paths(image_dataset_path,gap=2)
    print(vis_detection_result(graph,frame_lists[50],'/home/jiachang/vis_result.jpg'))
    # run_inference_for_images_per_image(graph,image_dataset_path,np_paths_boxes_path,0.4)
