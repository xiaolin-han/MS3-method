function [image_recon_3d] = MS3( HMS_train,HHS_train,Kb, Kc, HMS_test, nodes_num)
% inputs:
%     HMS_train: training HMS image m*n*\lambda_x;
%     HHS_train: training HHS image m*n*\lambda_Y
%     HMS_test: test HMS image M*N*\lambda_x;
%     Kb: the total number of blocks;
%     Kc: the total number ofsubspaces;
%     nodes_num: number of the second layer of Multi-branch BPNN, nodes_num = [10];
% output:
%     image_recon_3d: the reconstructed image.

image_trainx = reshape(HMS_train,[],4)';
image_trainy = reshape(HHS_train,[],size(HHS_train,3))';
image_testx =  reshape(HMS_test,[],4)';
% Modified Superpixel Segmentation
[L_slic,~]= superpixels(HMS_train,Kb,'Method', 'slic', 'Compactness' ,0.1);
C_L_2d = L_slic(:);
% typical spectra in each block are further divided into Kc subspaces 
Table_L = tabulate(C_L_2d);
Table_L_t = find(Table_L(:,2)~=0);
Table_L_left = Table_L(Table_L_t,:); 
clear spectrum_left
for i = 1: size(C_L_2d,1)  
    spectrum_t = find(C_L_2d==i);
    spectrum = image_trainx(:,spectrum_t);  
    spectrum_left(i,:) = mean(spectrum,2);
end  
[IDX_LK, C_LK, ~, ~] = kmeans([double(spectrum_left)], Kc);
labelall=zeros(1,length(C_L_2d));
for i = 1:Kc
    spectrumlk_t = find( IDX_LK==i );
    label_l = Table_L_left(spectrumlk_t,1);
    tt=[];
    for j = 1: length(spectrumlk_t)
        tt = [ tt; find(C_L_2d==label_l(j))];
    end  
    labelall(1,tt)=i; 
end
IDXkl_1d = labelall;

% test image
% Modified Superpixel Segmentation
[Lt_slic,~]= superpixels(HMS_test,Kb,'Method', 'slic', 'Compactness' ,0.1);
C_Lt_2d = Lt_slic(:);
% Supervised Clustering
Table_Lt = tabulate(C_Lt_2d);
Table_Lt_t = find(Table_Lt(:,2)~=0);
Table_Lt_left = Table_Lt(Table_Lt_t,:);
clear spectrumt_left 
for i = 1: size(Table_Lt_left,1)  
    clear spectrumt_t
    lableLt=Table_Lt_left(i,1);
    spectrumt_t = find(C_Lt_2d==lableLt);
    spectrumt = image_testx(:,spectrumt_t); 
    spectrumt_left(i,:) = mean(spectrumt,2); 
end
[ IDXt_LK ] = clusters_test_part( [double(spectrumt_left')], C_LK, Kc);
labelallt=zeros(1,length(C_Lt_2d));
for i = 1:Kc
    spectrumlkt_t = find( IDXt_LK==i );
    labelt_l = Table_Lt_left(spectrumlkt_t,1);
    tt=[];
    for j = 1: length(spectrumlkt_t)
        tt = [ tt; find(C_Lt_2d==labelt_l(j))];
    end  
%     spectrumall = image_trainy(:,tt);
    spectrumall(:,i) = mean(image_trainx(:,tt),2);
    labelallt(1,tt)=i; 
end
IDXkl_test = labelallt;

% training 
for i_class = 1: Kc
% data prepare 
index = [find(IDXkl_1d==i_class)];
classSAM_x1=image_trainx(:,index)';
classSAM_y1=image_trainy(:,index)';
index_test = [find(IDXkl_test==i_class)];
classSAM_x_test=image_testx(:,index_test)';
% Multi-branch BPNN training
input=classSAM_x1;
output=classSAM_y1;
k=rand(1,size(input,1));
[m,n]=sort(k);
input_train=input(n,:)';
output_train=output(n,:)';
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
% initialization
net=newff(inputn,outputn,nodes_num);
net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00004;
% training
net=train(net,inputn,outputn);
% Spectral Super-resolution
input_test=classSAM_x_test';
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
image_recon_clsteri=mapminmax('reverse',an,outputps);
% reconstruction for different subspaces
image_recon(:,index_test)=image_recon_clsteri; 
end
image_recon_3d = reshape(image_recon',size(HMS_test,1),size(HMS_test,2),[]);


