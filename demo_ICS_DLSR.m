% code is written by Jie Wen
% If any problems, please contact: wenjie@hrbeu.edu.cn
% Please cite the reference:
% Jie Wen, Yong Xu, Zuoyong Li, Zhongli Ma, Yuanrong Xu, 
% Inter-class sparsity based discriminative least square regression [J],
% Neural Networks, 2018, doi: 10.1016/j.neunet.2018.02.002.

clear all
clc
clear memory;

name = 'YaleB_32x32'
load (name);
fea = double(fea);
sele_num = 10;
nnClass = length(unique(gnd));  % The number of classes;
num_Class = [];
for i = 1:nnClass
  num_Class = [num_Class length(find(gnd==i))]; %The number of samples of each class
end
%%------------------select training samples and test samples--------------%%
Train_Ma  = [];
Train_Lab = [];
Test_Ma   = [];
Test_Lab  = [];
for j = 1:nnClass    
    idx      = find(gnd==j);
    randIdx  = randperm(num_Class(j));
    Train_Ma = [Train_Ma; fea(idx(randIdx(1:sele_num)),:)];            % select select_num samples per class for training
    Train_Lab= [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Ma  = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];  % select remaining samples per class for test
    Test_Lab = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
Train_Ma = Train_Ma';                       % transform to a sample per column
Train_Ma = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
Test_Ma  = Test_Ma';
Test_Ma  = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]);  % -------------

label = unique(Train_Lab);
Y = bsxfun(@eq, Train_Lab, label');
Y = double(Y)';

opts.miu = 1e-8;
opts.rho = 1.01;
opts.lambda1 = 1e-2;
opts.lambda2 = 5e-1;
opts.lambda3 = 1e-2;
opts.nnClass = nnClass;
opts.maxIter = 30;
X = Train_Ma;

[Q,E,obj] = ICS_DLSR(X,Y,Train_Lab,opts);
% % figure;plot(obj);
Train_Maa = Q*Train_Ma;
Test_Maa  = Q*Test_Ma;
Train_Maa = Train_Maa./repmat(sqrt(sum(Train_Maa.^2)),[size(Train_Maa,1) 1]);
Test_Maa  = Test_Maa./repmat(sqrt(sum(Test_Maa.^2)),[size(Test_Maa,1) 1]);    
[class_test] = knnclassify(Test_Maa', Train_Maa', Train_Lab,1,'euclidean','nearest');
rate_acc = sum(Test_Lab == class_test)/length(Test_Lab)*100
