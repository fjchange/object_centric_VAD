import numpy as np
import cv2
from PIL import Image
import os.path as osp
import os
# import tensorflow as tf
from scipy.signal import savgol_filter
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import normalize

image_size_map={
    'avenue':(360,640),
    'shanghaitech':(480,856),
    'ped1':(158,238),
    'ped2':(240,360),
    'umn':(240,320)
}

def data_preprocessing(img_path,target_size=None):
    img=Image.open(img_path)
    (im_width, im_height) = img.size
    img=np.array(img).reshape((im_height, im_width, 3)).astype(np.uint8)
    if target_size!=None:
        img=cv2.resize(img,(target_size,target_size))
    return img

def get_frames_paths(image_folder,gap):
    vid_lists=sorted([osp.join(image_folder,path) for path in os.listdir(image_folder)])
    frame_paths=[]
    for vid in vid_lists:
        temp_paths=[osp.join(vid,path) for path in sorted(os.listdir(vid))]
        frame_paths.extend(temp_paths[gap:-gap])

    return frame_paths

def get_vids_paths(image_folder):
    vid_lists=sorted([osp.join(image_folder,path) for path in os.listdir(image_folder)])
    frame_paths=[]
    for vid in vid_lists:
        frame_paths.append([osp.join(vid,path) for path in sorted(os.listdir(vid))])
    return frame_paths


def box_image_crop(image_path,box,target_size=64):
    image=cv2.imread(image_path,0)

    box=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
    crop_image=image[box[0]:box[2],box[1]:box[3]]
    crop_image=cv2.resize(crop_image,dsize=(target_size,target_size))
    crop_image=np.array(crop_image).reshape((target_size, target_size, 1)).astype(np.float32)/255.0

    return crop_image

def split_path_boxes(prefix,path_box_list,dataset_name,img_height,img_width):
    paths_former=[]
    paths_gray=[]
    paths_back=[]

    boxes=[]
    class_indexes=[]

    for item in path_box_list:
        #paths_gray.append(item[0])

        path_list=item[0].split('/')
        temp_path=prefix

        for i in range(3,len(path_list)-1):
            temp_path=osp.join(temp_path,path_list[i])
        paths_gray.append(osp.join(temp_path,path_list[-1]))

        path_prefix=temp_path
        if dataset_name!='shanghaitech':
            index=int(path_list[-1].split('.')[0])
            if len(path_list[-1].split('.')[0])==4:
                paths_former.append(path_prefix+'/'+'%04d'%(index-2)+'.jpg')
                paths_back.append(path_prefix+'/'+'%04d'%(index+2)+'.jpg')
            else:
                paths_former.append(path_prefix+'/'+'%03d'%(index-2)+'.jpg')
                paths_back.append(path_prefix+'/'+'%03d'%(index+2)+'.jpg')
        else:
            index_list=path_list[-1].split('.')[0].split('_')
            index=int(index_list[-1])
            paths_former.append(path_prefix+'/'+index_list[0]+'_'+index_list[1]+'_'+'%04d'%(index-2)+'.jpg')
            paths_back.append(path_prefix+'/'+index_list[0]+'_'+index_list[1]+'_'+'%04d'%(index+2)+'.jpg')

        boxes.append([int(float(item[1])*img_height),int(float(item[2])*img_width),
                      int(img_height*(float(item[3])-float(item[1]))),int(img_width*(float(item[4])-float(item[2])))])
        if len(item)==6:
            class_indexes.append(int(item[-1]))
    return paths_former,paths_gray,paths_back,boxes,class_indexes

# def _crop_img(path,box,target_size):
#     file_content=tf.read_file(path)
#     jpg=tf.image.decode_and_crop_jpeg(file_content,crop_window=box,channels=3)
#     jpg=tf.image.resize_images(jpg,size=(target_size,target_size))
#     jpg=tf.image.rgb_to_grayscale(jpg)
#     jpg=tf.cast(jpg,dtype=tf.float32)
#     return jpg

def Conv_AE_dataset(image_folder,gray=True,target_size=227):
    vids=get_vids_paths(image_folder)
    frames=[]
    for vid in vids:
        _temp_frames=[]
        for frame_path in vid:
            if gray:
                _temp_frames.append(cv2.resize(cv2.imread(frame_path,0),(target_size,target_size)).astype(np.float32))
            else:
                _temp_frames.append(cv2.resize(cv2.imread(frame_path)[...,::-1],(target_size,target_size)).astype(np.float32))
        frames.append(_temp_frames)
    return frames


def CAE_dataset_feed_dict(prefix,np_path_box,dataset_name):
    path_box_list=np.load(np_path_box)
    (image_height, image_width) = image_size_map[dataset_name]
    former_paths,gray_paths,back_paths,boxes,class_indexes=split_path_boxes(prefix,path_box_list,dataset_name,image_height,image_width)

    f_imgs=[]
    g_imgs=[]
    b_imgs=[]
    for f_path,g_path,b_path,box in zip(former_paths,gray_paths,back_paths,boxes):
        f_imgs.append(box_image_crop(f_path,box))
        g_imgs.append(box_image_crop(g_path,box))
        b_imgs.append(box_image_crop(b_path,box))

    return f_imgs,g_imgs,b_imgs,class_indexes

# def score_smoothing(score):
#     score_len=score.shape[0]//9
#     if score_len%2==0:
#         score_len+=1
#     score=savgol_filter(score,score_len,3)
#
#     return score
#
def score_smoothing(score,sigma=30):
    # r = score.shape[0] //39
    # if r%2==0:
    #     r+=1
    r = 125
    if r > score.shape[0] // 2:
        r = score.shape[0] // 2 - 1
    if r % 2 == 0:
        r += 1
    gaussian_temp=np.ones(r*2-1)
    for i in range(r*2-1):
        gaussian_temp[i]=np.exp(-(i-r)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    new_score=score
    for i in range(r,score.shape[0]-r):
        new_score[i]=np.dot(score[i-r:i+r-1],gaussian_temp)
    return new_score

def _log10(a):
    numerator=np.log(a)
    denominator=np.log(10)
    return numerator/denominator

def cal_psnr(img_pred,img_gt):
    # the img is [0,1]
    shape=img_pred.shape
    num_pixels=float(shape[0]*shape[1]*shape[2])
    square_diff=np.square(img_pred-img_gt+1e-8)
    psnr=10*_log10(1/((1/num_pixels)*np.sum(square_diff)))
    return psnr

def l2_err(img_pred,img_gt):
    return np.mean(np.square(img_pred-img_gt+1e-8))

def norm_(feat,l=1):
    return normalize(feat,l)


