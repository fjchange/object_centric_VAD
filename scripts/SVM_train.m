% This Code used the clustered features to train the SVM, num of the SVMs is 10
% data.mat is [num_features,features_len], labels.mat is [K,num_features]

data_path='../matlab_files/data.mat'
labels_path='../matlab_files/labels.mat'
weights_path='../matlab_files/weights.mat'
biases_path='../matlab_files/biases.mat'

K=10

data=load(data_path)
data=reshape(data.data,3072,[])
labels=load(labels_path)
labels=labels.labels


% Every row of labels is a binary embedding
for ii=1:size(labels,1)
    label=labels(ii,:)
    label=double(label)
    [w,b,info]=vl_svmtrain(data,label,1)

    if ii==1
        W=w
        B=b
    else
        W=cat(2,W,w)
        B=cat(2,B,b)
    end
end
save(weights_path,'W')
save(biases_path,'B')

fprintf('train SVM finished!')
