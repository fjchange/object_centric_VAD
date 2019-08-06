import numpy as np
import cv2
from PIL import Image
import os.path as osp
import os
import tensorflow as tf



def data_preprocessing(img_path):
    img=Image.open(img_path)
    (im_width, im_height) = img.size
    return np.array(img).reshape((im_height, im_width, 3)).astype(np.uint8)

def get_frames_paths(image_folder,gap):
    vid_lists=sorted([osp.join(image_folder,path) for path in os.listdir(image_folder)])
    frame_paths=[]
    for vid in vid_lists:
        temp_paths=[osp.join(vid,path) for path in sorted(os.listdir(vid))]
        frame_paths.extend(temp_paths[gap:-gap])

    return frame_paths

def get_vids_paths(image_folder,gap):
    vid_lists=sorted([osp.join(image_folder,path) for path in os.listdir(image_folder)])
    frame_paths=[]
    for vid in vid_lists:
        frame_paths.append([osp.join(vid,path) for path in sorted(os.listdir(vid))])
    return frame_paths

rgb2xyz = (0.412453,0.357580, 0.180423, 0,
           0.212671,0.715160, 0.072169, 0,
           0.019334,0.119193, 0.950227, 0 )
def box_image_crop(image_path,box):
    image=Image.open(image_path)
    image = image.convert("L", rgb2xyz)
    image_width,image_height=image.size
    image=np.array(image).reshape((image_height, image_width, 1)).astype(np.uint8)

    box=[int(box[0]*image_width),int(box[1]*image_height),int(box[2]*image_width),int(box[3]*image_height)]
    crop_image=image[box[0]:box[2],box[1]:box[3],:]
    crop_image=crop_image.astype(np.float32)
    return crop_image

def split_path_boxes(path_box_list,dataset_name,img_height,img_width):
    paths_former=[]
    paths_gray=[]
    paths_back=[]

    boxes=[]

    for item in path_box_list:
        paths_gray.append(item[0])

        path_list=item[0].split('/')
        temp_path=path_list[0]
        for i in range(1,len(path_list)-1):
            temp_path=osp.join(temp_path,path_list[i])
        path_prefix=temp_path
        if dataset_name!='shanghaitech':
            index=int(path_list[-1].split('.')[0])
            paths_former.append(path_prefix+'/'+'%04d'%(index-2)+'.jpg')
            paths_back.append(path_prefix+'/'+'%04d'%(index+2)+'.jpg')
        else:
            index_list=path_list[-1].split('.')[0].split('_')
            index=int(index_list[-1])
            paths_former.append(path_prefix+'/'+index_list[0]+'_'+index_list[1]+'_'+'%04d'%(index-2)+'.jpg')
            paths_back.append(path_prefix+'/'+index_list[0]+'_'+index_list[1]+'_'+'%04d'%(index+2)+'.jpg')

        boxes.append([int(float(item[1])*img_height),int(float(item[2])*img_width),
                      int(img_height*(float(item[3])-float(item[1]))),int(img_width*(float(item[4])-float(item[2])))])
    return paths_former,paths_gray,paths_back,boxes

def _crop_img(path,box,target_size):
    file_content=tf.read_file(path)
    jpg=tf.image.decode_and_crop_jpeg(file_content,crop_window=box,channels=3)
    jpg=tf.image.resize_images(jpg,size=(target_size,target_size))
    jpg=tf.image.rgb_to_grayscale(jpg)
    jpg=tf.cast(jpg,dtype=tf.float32)
    return jpg

def CAE_dataset(np_path_box,dataset_name,num_epochs,batch_size):
    path_box_list=np.load(np_path_box)
    (image_width, image_height) = Image.open(path_box_list[0][0]).size
    former_paths,gray_paths,back_paths,boxes=split_path_boxes(path_box_list,dataset_name,image_height,image_width)
    # boxes=np.array(boxes,dtype=np.int32)

    former_paths_tensor=tf.convert_to_tensor(former_paths,dtype=tf.string)
    gray_paths_tensor=tf.convert_to_tensor(gray_paths,dtype=tf.string)
    back_paths_tensor=tf.convert_to_tensor(back_paths,dtype=tf.string)

    boxes_tensor=tf.convert_to_tensor(boxes,dtype=tf.int32)

    input_queue=tf.train.slice_input_producer([former_paths_tensor,gray_paths_tensor,back_paths_tensor,boxes_tensor],num_epochs=num_epochs,shuffle=True)

    box_tensor=tf.identity(input_queue[-1])

    crop_img_1=_crop_img(input_queue[0],box_tensor,64)
    crop_img_2=_crop_img(input_queue[1],box_tensor,64)
    crop_img_3=_crop_img(input_queue[2],box_tensor,64)

    grad1_x,grad1_y=tf.image.image_gradients(tf.expand_dims(crop_img_1,axis=0))
    grad2_x,grad2_y=tf.image.image_gradients(tf.expand_dims(crop_img_2,axis=0))
    grad3_x,grad3_y=tf.image.image_gradients(tf.expand_dims(crop_img_3,axis=0))

    grad_dis_1=tf.sqrt(tf.square(grad2_x-grad1_x)+tf.square(grad2_y-grad1_y))
    grad_dis_2=tf.sqrt(tf.square(grad3_x-grad2_x)+tf.square(grad3_y-grad2_y))

    grad_dis_1=tf.reshape(grad_dis_1,shape=[grad_dis_1.shape[1],grad_dis_1.shape[2],grad_dis_1.shape[3]])
    grad_dis_2=tf.reshape(grad_dis_2,shape=[grad_dis_2.shape[1],grad_dis_2.shape[2],grad_dis_2.shape[3]])

    gray_batch,former_batch,back_batch=tf.train.batch([crop_img_2,grad_dis_1,grad_dis_2],batch_size=batch_size,num_threads=18)
    return former_batch,gray_batch,back_batch


def CAE_dataset_feed_dict(np_path_box,dataset_name):
    path_box_list=np.load(np_path_box)
    (image_width, image_height) = Image.open(path_box_list[0][0]).size
    former_paths,gray_paths,back_paths,boxes=split_path_boxes(path_box_list,dataset_name,image_height,image_width)

    f_imgs=[]
    g_imgs=[]
    b_imgs=[]
    for f_path,g_path,b_path,box in zip(former_paths,gray_paths,back_paths,boxes):
        f_imgs.append(box_image_crop(f_path,box))
        g_imgs.append(box_image_crop(g_path,box))
        b_imgs.append(box_image_crop(b_path,box))

    return f_imgs,g_imgs,b_imgs


