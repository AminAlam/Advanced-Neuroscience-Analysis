%% %%%%%%%%%%%%%%% part 1 -Simulate sparse basis functions of the natural images
clc
clear
close all
addpath 'sparsenet/'

clc
close all
load data/IMAGES.mat
load data/IMAGES_RAW.mat

%% showing the images

for img_index = 1:size(IMAGESr, 3)
    % raw images
    figure
    img = IMAGESr(:,:,img_index)-min(IMAGESr(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    imshow(img);
    colormap gray
    % whitened images
    figure
    img = IMAGES(:,:,img_index)-min(IMAGES(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    imshow(img);
    colormap gray
end
%% showing basis functions
clc
close all
A = rand(64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet
%% %%%%%%%%%%%%%%% part 2 - Study the effect of different datasets

% %%%%%% yale
clc
close all
IMAGES = zeros(192,168,10);

for i = 1:10
    if i <10
        img = imread("data/CroppedYale/yaleB0"+num2str(i)+"/yaleB0"+num2str(i)+"_P00A+000E+00.pgm");
    else
        img = imread("data/CroppedYale/yaleB"+num2str(i)+"/yaleB"+num2str(i)+"_P00A+000E+00.pgm");
    end
    IMAGES(:,:,i) = img;
end
IMAGESr = IMAGES;
% whitening images
for img_index = 1:size(IMAGES, 3)
    img = IMAGES(:,:,img_index);
    img = img/max(img,[],'all');  
    img = img-mean(img, 'all');
    IMAGES_tmp(:,:,img_index) = img(13:end-12,:);
end
IMAGES = IMAGES_tmp;
IMAGES = whitener(IMAGES);
% showing the images
for img_index = 1:size(IMAGESr, 3)
    % raw images
    figure
    img = IMAGESr(:,:,img_index)-min(IMAGESr(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    imshow(img);
    colormap gray
    % whitened images
    figure
    img = IMAGES(:,:,img_index)-min(IMAGES(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    imshow(img);
    colormap gray
end

% showing basis functions

A = rand(64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet

%% %%%%%% MNIST
clc
close all
IMAGES = load('data/mnist.mat');
IMAGES = IMAGES.test.images;
IMAGES = IMAGES(:,:,1:10);
IMAGESr = IMAGES;
% whitening images
for img_index = 1:size(IMAGES, 3)
    img = IMAGES(:,:,img_index);
    img = img/max(img,[],'all');  
    img = img-mean(img, 'all');
    IMAGES(:,:,img_index) = img;
end
IMAGES = whitener(IMAGES);
% showing the images
for img_index = 1:size(IMAGESr, 3)
    % raw images
    figure
    img = IMAGESr(:,:,img_index)-min(IMAGESr(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    imshow(img);
    colormap gray
    % whitened images
    figure
    img = IMAGES(:,:,img_index)-min(IMAGES(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    imshow(img);
    colormap gray
end
% showing basis functions

A = rand(64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet