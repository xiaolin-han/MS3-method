
% inputs:
%     HMS_train: training HMS image m*n*\lambda_x,  \lambda_x =4 bands;
%     HHS_train: training HHS image m*n*\lambda_Y
%     HMS_test: test HMS image M*N*\lambda_x,   \lambda_x =4 bands;
%     Kb: the total number of blocks, Kb=7000
%     Kc: the total number ofsubspaces, Kc = 20
%     nodes_num: number of the second layer of Multi-branch BPNN, nodes_num = [10];
% output:
%     image_recon_3d: the reconstructed image
clc;
clear all;
warning off;
addpath(genpath(pwd));
load HMS_train 
load HHS_train
load HMS_test
nodes_num =10;
Kb = 7000;
Kc = 20;
[image_recon_3d] = MS3( HMS_train,HHS_train,Kb, Kc, HMS_test, nodes_num);

