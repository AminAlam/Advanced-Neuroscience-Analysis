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
montage(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
montage(IMAGES_cell,'size',[2,5])
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
montage(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
montage(IMAGES_cell,'size',[2,5])
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
montage(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
montage(IMAGES_cell,'size',[2,5])
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
montage(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
montage(IMAGES_cell,'size',[2,5])
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
    frames_n(:,:,:,frame_no) = frames(:, 33:end-32,:,frame_no);
    frame = whitener(frame);
    frames_prep(:,:,frame_no) = frame;
end
frames = frames_n;

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
montage(IMAGES_cell_raw,'size',[2,5])
colormap gray
figure
montage(IMAGES_cell,'size',[2,5])
colormap gray

%% learning the basis functions
clc
close all

IMAGES = frames_prep(:,:,1:10);

A = rand(256, 100)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet

%% calculating coefficients for each frame
clc
close all

imSz = size(frames_prep,[1 2]);
patchSz = [sz sz];
xIdxs = [1:patchSz(2):imSz(2) imSz(2)+1];
yIdxs = [1:patchSz(1):imSz(1) imSz(1)+1];

batch_size = (length(yIdxs)-1)^2;

S_all = zeros(size(A,2), batch_size, size(frames, 3));

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
    S_all(:,:,frame_no) = S;
end

%% plotting the basis function which has the bigesst coeff for each of the patchs
clc
close all
figure(1)
figure(2)

[L M]=size(A);
sz=sqrt(L);
buf=1;
if floor(sqrt(M))^2 ~= M
  m=sqrt(M/2);
  n=M/m;
else
  m=sqrt(M);
  n=m;
end

for frame_no = 11:size(frames_prep, 3)
    frame_no
    S = S_all(:,:,frame_no);
    array_all=-ones(1+m*(sz+1),1+n*(sz+1));
    [max_coeffs_all, max_coeffs_index_all] = maxk(S, 100);
        
    for max_k = 30
        array=-ones(1+m*(sz+1),1+n*(sz+1));
        max_coeffs_index = max_coeffs_index_all(max_k,:);
        max_coeffs = max_coeffs_all(max_k,:);
        max_coeffs_index_vector = max_coeffs_index;
        max_coeffs_vector = max_coeffs;

        basis_function_map = {};
        for i = 1:length(max_coeffs_index_vector)
            basis_function = return_basis_functions(A,S,max_coeffs_index_vector(i));
            basis_function_map{i} = basis_function*max_coeffs_vector(i);
        end

        k=1;
        for i=1:m
            for j=1:n
                array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz])=...
                basis_function_map{k};
                k=k+1;
            end
        end
        array_all = array_all+array;
    end    
    
    figure(1)
    img_imagesc = imagesc(array_all);
    set(gca,'YDir','normal')
    caxis([1, size(A,2)])
    img_imagesc = img_imagesc.CData;
    img_imagesc = imresize(img_imagesc, [288, 288]);

    img_main = frames(:,:,:,frame_no);
    image(img_main);
    
    hold on
    img_imagesc = imagesc(img_imagesc);
    img_imagesc.AlphaData = 0.4;
    colormap jet
    caxis([-2, 2])
    hold off
    
    figure(2)
    imagesc(array_all);
    colormap jet
    caxis([-2, 2])
    pause(0.5)
end

%% patch plots
clc
close all

[L M]=size(A);
sz=sqrt(L);
buf=1;
if floor(sqrt(M))^2 ~= M
  m=sqrt(M/2);
  n=M/m;
else
  m=sqrt(M);
  n=m;
end

sudden_moves = [38, 60, 80, 88];
patchs = [1, 15, 30, 45, 80, 100];
for patch_no = patchs
    figure
    for sudden_move = sudden_moves
        xline(sudden_move, '--k');
    end
    for basis_no = 1:20
        hold on
        basis_values = S_all(basis_no, patch_no, 11:end);
        plot(11:size(frames,4), squeeze(basis_no*2+basis_values), 'LineWidth', 2)
    end
    xlim([11, size(frames,4)])
    xlabel('Frames')
    ylabel('Basis Functions')
    title("Patch No."+num2str(patch_no))
    legend('Sudden Moves')
    
    figure
    array=zeros(1+m*(sz+1),1+n*(sz+1));
    k=1;
    for i=1:m
        for j=1:n
            if k == patch_no
                array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz])=1+...
                    array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz]);
            end
            k=k+1;
        end
    end
    
    img_imagesc = imagesc(array);
    set(gca,'YDir','normal')
    caxis([0, 1])
    img_imagesc = img_imagesc.CData;
    img_imagesc = imresize(img_imagesc, [288, 288]);

    img_main = frames(:,:,:,1);
    image(img_main);
    
    hold on
    img_imagesc = imagesc(img_imagesc);
    img_imagesc.AlphaData = 0.5;
    hold off 
end

%% %%%%%%%%%%%%%%% the role of attention models in the basis functions
addpath 'gbvs/'
clc
close all

