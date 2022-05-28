
clc;
clear all;
load('TEST.mat')
X = X(1:10);
Y = Y(1:10);


score = rocScoreSaliencyVsFixations(salmap,X,Y,origimgsize)




