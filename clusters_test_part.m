
function [ pattern ] = clusters_test_part( image_testx, center, N)
% Supervised Clustering
[m,n]=size(image_testx');
pattern=zeros(m,1);
distence=zeros(1,N);
for x=1:m
    for y=1:N
    distence(y)=norm(image_testx(:,x)'-center(y,:));
    end
    [~, temp]=min(distence);
    pattern(x,1)=temp; 
end
