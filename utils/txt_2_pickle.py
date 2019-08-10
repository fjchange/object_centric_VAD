import pickle
import numpy as np
import csv
import os

def read_txt_2_numpy(path):
    #print(path)

    a=[]
    with open(path)as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            for item in row:
                a.append(float(item))
    print(a.__len__())
    a=np.array(a,dtype=np.float32)
    return a

def get_file_and_sort(dir,dataset):
    vids=[os.path.join(dir,path) for path in sorted(os.listdir(dir))]
    a=[]
    for vid in vids:
        a.append(read_txt_2_numpy(vid))
    # pickle_path=dir+'.pkl'
    #
    # result_dict = {'dataset': dataset, 'psnr': a, 'flow': [], 'names': [],
    #                'diff_mask': []}
    #
    # with open(pickle_path, 'wb') as writer:
    #     pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    get_file_and_sort('/home/manning/autor_results/Ionescu_et_al_CVPR_2019_ShanghaiTech_results','shanghaitech')