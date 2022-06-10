%% %%%%%%%%%%%%%%% part 1 -Simulate sparse basis functions of the natural images
clc
clear
close all
addpath 'sparsenet/'

clc
close all
load data/IMAGES.mat
load data/IMAGES_RAW.mat

% showing the images
IMAGES_cell_raw = {};
IMAGES_cell = {};
for img_index = 1:size(IMAGESr, 3)
    % raw images
    img = IMAGESr(:,:,img_index)-min(IMAGESr(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    IMAGES_cell_raw{img_index} = img;
    % whitened images
    img = IMAGES(:,:,img_index)-min(IMAGES(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    IMAGES_cell{img_index} = img;
end
figure
MONTAGE(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
MONTAGE(IMAGES_cell,'size',[2,5])
colormap gray

% showing basis functions
clc
A = rand(256, 64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet
%% %%%%%%%%%%%%%%% part 2 - Study the effect of different datasets

% %%%%%% yale
clc
close all
IMAGES = zeros(192,168,10);

IMAGES_cell_raw = {};
IMAGES_cell = {};

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
    img = IMAGESr(:,:,img_index)-min(IMAGESr(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    IMAGES_cell_raw{img_index} = img;
    % whitened images
    img = IMAGES(:,:,img_index)-min(IMAGES(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    IMAGES_cell{img_index} = img;
end
figure
MONTAGE(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
MONTAGE(IMAGES_cell,'size',[2,5])
colormap gray

% showing basis functions
A = rand(256)-0.5;
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

IMAGES_cell_raw = {};
IMAGES_cell = {};

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
    img = IMAGESr(:,:,img_index)-min(IMAGESr(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    IMAGES_cell_raw{img_index} = img;
    % whitened images
    img = IMAGES(:,:,img_index)-min(IMAGES(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    IMAGES_cell{img_index} = img;
end
figure
MONTAGE(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
MONTAGE(IMAGES_cell,'size',[2,5])
colormap gray

% showing basis functions
A = rand(64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet

%% %%%%%% caltech 110
clc
close all
IMAGES = zeros(140,140,10);

IMAGES_cell_raw = {};
IMAGES_cell = {};

for i = 1:10
    if i <10
        img = imread("data/caltech_101/image_"+num2str(i)+".jpeg");
    end
    if size(img, 3)>1
        img = rgb2gray(img);
    end
    img = imresize(img, [140, 140]);
    IMAGES(:,:,i) = img;
end
IMAGESr = IMAGES;
% whitening images
for img_index = 1:size(IMAGES, 3)
    img = IMAGES(:,:,img_index);
    img = img/max(img,[],'all');  
    img = img-mean(img, 'all');
    IMAGES(:,:,img_index) = img;
end
IMAGES = whitener(IMAGES);

for img_index = 1:size(IMAGESr, 3)
    % raw images
    img = IMAGESr(:,:,img_index)-min(IMAGESr(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    IMAGES_cell_raw{img_index} = img;
    % whitened images
    img = IMAGES(:,:,img_index)-min(IMAGES(:,:,img_index),[],'all');
    img = img*255/max(img,[],'all');
    IMAGES_cell{img_index} = img;
end
figure
MONTAGE(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
MONTAGE(IMAGES_cell,'size',[2,5])
colormap gray

% showing basis functions
A = rand(64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet

%% %%%%%%%%%%%%%%% part 3 - Study the dynamics of the sparse coefficients
clc
close all
video = VideoReader('data/BIRD.avi');
frames = read(video,[1 Inf]);

IMAGES_cell_raw = {};
IMAGES_cell = {};

% preprocessing frames of the video
for frame_no = 1:size(frames, 4)
    frame = frames(:,:,:,frame_no);
    frame = rgb2gray(frame);
    frame = frame(:, 33:end-32);
    frame = whitener(frame);
    frames_prep(:,:,frame_no) = frame;
end


% showing the frames of the video
for frame_no = 1:10
    frame = frames(:,:,:,frame_no);
    frame_prep = frames_prep(:,:,frame_no);
    frame_prep = frame_prep-min(frame_prep,[],'all');
    frame_prep = frame_prep*255/max(frame_prep,[],'all');
    IMAGES_cell_raw{frame_no} = frame;
    IMAGES_cell{frame_no} = frame_prep;
end

figure
MONTAGE(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
MONTAGE(IMAGES_cell,'size',[2,5])
colormap gray

% learning the basis functions

IMAGES = frames_prep(:,:,1:10);

A = rand(256, 100)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet

%%
clc
imSz = size(frames_prep,[1 2]);
patchSz = [sz sz];
xIdxs = [1:patchSz(2):imSz(2) imSz(2)+1];
yIdxs = [1:patchSz(1):imSz(1) imSz(1)+1];

batch_size = (length(yIdxs)-1)^2;

S_all = {};
figure
for frame_no = 11:size(frames_prep, 3)
    I = frames_prep(:,:,frame_no);
    X = zeros(L,batch_size);
    patch_count = 1;
    
    for i = 1:length(yIdxs)-1
        Isub = I(yIdxs(i):yIdxs(i+1)-1,:);
        for j = 1:length(xIdxs)-1
            X(:,patch_count)=reshape(Isub(:,xIdxs(j):xIdxs(j+1)-1),L,1);
            patch_count = patch_count+1;
        end
    end

    S = cgf_fitS(A,X,noise_var,beta,sigma,tol);
    S_all{frame_no} = S;
    [~,max_coeffs] = max(S);
    textStrings = num2str(max_coeffs');
    textStrings = strtrim(cellstr(textStrings));
    max_coeffs_vector = max_coeffs;
    max_coeffs = reshape(max_coeffs, [sqrt(batch_size), sqrt(batch_size)]);

    imagesc(max_coeffs)
    set(gca,'YDir','normal')

    [x, y] = meshgrid(1:sqrt(batch_size), 1:sqrt(batch_size));  
    hStrings = text(x(:), y(:), textStrings(:), ...  
                    'HorizontalAlignment', 'center');
    midValue = mean(get(gca, 'CLim'));  
    textColors = repmat(max_coeffs_vector(:) > midValue, 1, 3);  
    textColors = ~textColors;
    set(hStrings, {'Color'}, num2cell(textColors, 2));
    caxis([1, size(A,2)])
    colormap gray
    colorbar
    pause(0.5)
end


