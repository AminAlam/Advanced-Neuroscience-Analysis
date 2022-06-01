%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Eye Tracking Database
clc
clear
close all
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/Eye tracking database/DatabaseCode/';

%% single subject
clc
close all
datafolder = 'Eye tracking database/DATA/hp';
stimfolder = 'Eye tracking database/ALLSTIMULI';
showEyeData(datafolder, stimfolder)

%% multiple subjects
clc
close all
stimfolder = 'Eye tracking database/ALLSTIMULI';
datafolder_all = 'Eye tracking database/DATA';
numFix = 5;
showEyeDataAcrossUsers(datafolder_all, stimfolder, numFix)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Saliency Model
clc
clear
close all
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/saliency model/JuddSaliencyModel';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/saliency model';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/matlabPyrTools/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/SaliencyToolbox/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/voc-release5/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/voc-release5/features/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/FaceDetect/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/features/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/imagemanipulation/';


datafolder = "Eye tracking database/DATA/hp";
files=dir(fullfile(datafolder,'*.mat'));
[filenames{1:size(files,1)}] = deal(files.name);

file_number = 940;

imagefile = filenames(file_number);
imagefile = imagefile{1};
imagefile = imagefile(1:end-4);
file_name = imagefile;
datafolder = "Eye tracking database/";
imagefile = datafolder+"ALLSTIMULI/"+imagefile+".jpeg";
FEATURES = saliency(imagefile);

load model;

subject_name = "hp";
datafolder = "Eye tracking database/";
data_file = datafolder+"DATA/"+subject_name+"/"+file_name+".mat";
data = load(data_file);
data = data.(sprintf("%s", file_name)).DATA.eyeData;

X_first_half = data(1:floor(end/2), 1);
Y_first_half = data(1:floor(end/2), 2);

X_second_half = data(floor(end/2)+1:end, 1);
Y_second_half = data(floor(end/2)+1:end, 2);

img = imread(imagefile);
[w_img, h, c] = size(img);
dims = [200, 200];
% map{1} = img;

figure
imshow(img)
hold on
plot(X_first_half, Y_first_half, '.y')
scatter(X_first_half(1), Y_first_half(1), 'g', 'filled')
scatter(X_first_half(end), Y_first_half(end), 'r', 'filled')
colormap gray
set(gca,'xtick',[])
set(gca,'ytick',[])

save_figure("Report/images/saliency/only_img"+num2str(file_number))

% whiten the feature data with the parameters from the model.
meanVec = model.whiteningParams(1, :);
stdVec = model.whiteningParams(2, :);
FEATURES=FEATURES-repmat(meanVec, [size(FEATURES, 1), 1]);
FEATURES=FEATURES./repmat(stdVec, [size(FEATURES, 1), 1]);

for type = 1:8
    w = model.w;
    if (type==1) %subband
        w(:, 1:12) = 0;
        w(:, 14:end) = 0;
    elseif (type==2) %Itti
        w(:, 1:13) = 0;
        w(:, 17:end) = 0;
    elseif (type==3) %Color
        w(1:16) = 0;
        w(:, 28:end) = 0;
    elseif type==4 %Torralba
        w(:, 1:27) = 0;
        w(:, 29:end) = 0;
    elseif type==5 %Horizon
        w(:, 1:28) = 0;
        w(:, 30:end) = 0;
    elseif type==6 %Object
        w(:, 1:29) = 0;
        w(:, 32:end) = 0;
    elseif type==7 %Center
        w(:, 1:32) = 0;
    elseif type==8 %all
        w = model.w;
    end


    % find the saliency map given the features and the model
    saliencyMap = (FEATURES*w(1:end-1)') + w(end);
    saliencyMap = (saliencyMap-min(saliencyMap))/(max(saliencyMap)-min(saliencyMap));
    saliencyMap = reshape(saliencyMap, dims);
    saliencyMap = imresize(saliencyMap, [w_img, h]);
    figure
    imshow(saliencyMap*255/max(saliencyMap, [], 'all'))
    colormap gray;
    
    hold on
    plot(X_first_half, Y_first_half, '.y')
    scatter(X_first_half(1), Y_first_half(1), 'g', 'filled')
    scatter(X_first_half(end), Y_first_half(end), 'r', 'filled')
    colormap gray
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    
    save_figure("Report/images/saliency/only_img"+num2str(file_number)+"_"+num2str(type))
end

%% showing eye positoin on saliency map

clc
clear
close all
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/Eye tracking database';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/saliency model/JuddSaliencyModel';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/saliency model';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/matlabPyrTools/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/SaliencyToolbox/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/voc-release5/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/voc-release5/features/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/FaceDetect/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/features/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/imagemanipulation/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/gbvs/gbvs';

% demo for one image
load model;

datafolder = "Eye tracking database/DATA/hp";
files=dir(fullfile(datafolder,'*.mat'));
[filenames{1:size(files,1)}] = deal(files.name);

subject_names = {"CNG", "ajs", "emb", "ems", "ff", "hp", "jcw", "jw", "kae", "krl", "po", "tmj", "tu", "ya", "zb"};
filenames = filenames(11:end);

subject_name = subject_names(1);
 
file_counter = 0;
for file_name = filenames
    file_counter = file_counter+1;
    file_name = file_name{1};
    file_name = file_name(1:end-4);

    datafolder = "Eye tracking database/";
    data_file = datafolder+"DATA/"+subject_name+"/"+file_name+".mat";
    data = load(data_file);
    data = data.(sprintf("%s", file_name)).DATA.eyeData;

    img_name = datafolder+"ALLSTIMULI/"+file_name+".jpeg";
    fixation_map_name = datafolder+"ALLFIXATIONMAPS/"+file_name+"_fixMap"+".jpg";

    datafolder = "saliency model/SaliencyMaps/";
    saliency_map_name = datafolder+file_name+"SM"+".jpg";

    img = imread(img_name);
    FEATURES_all = saliency(img_name);
    fixation_map = imread(fixation_map_name);

    [w_img, h, c] = size(img);
    dims = [200, 200];
    map{1} = img;

    origimgsize = size(img);
    origimgsize = origimgsize(1:2);
    
    X_first_half = data(1:floor(end/2), 1);
    Y_first_half = data(1:floor(end/2), 2);

    X_second_half = data(floor(end/2)+1:end, 1);
    Y_second_half = data(floor(end/2)+1:end, 2);

    FEATURES = FEATURES_all;

    % whiten the feature data with the parameters from the model.
    meanVec = model.whiteningParams(1, :);
    stdVec = model.whiteningParams(2, :);
    FEATURES=FEATURES-repmat(meanVec, [size(FEATURES, 1), 1]);
    FEATURES=FEATURES./repmat(stdVec, [size(FEATURES, 1), 1]);

    % find the saliency map given the features and the model
    saliencyMap = (FEATURES*model.w(1:end-1)') + model.w(end);
    saliencyMap = (saliencyMap-min(saliencyMap))/(max(saliencyMap)-min(saliencyMap));
    saliencyMap = reshape(saliencyMap, dims);
    saliencyMap = imresize(saliencyMap, [w_img, h]);
    
    figure
    imshow(img)
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    
    save_figure("Report/images/saliency/img"+num2str(file_counter))
    
    figure
    imshow(255*saliencyMap/max(saliencyMap, [], 'all'))
    hold on
    plot(X_first_half, Y_first_half, '.y')
    scatter(X_first_half(1), Y_first_half(1), 'g', 'filled')
    scatter(X_first_half(end), Y_first_half(end), 'r', 'filled')
    colormap gray
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    save_figure("Report/images/saliency/img"+num2str(file_counter)+"_saliency")
    pause(0.2)
    
    close all
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% comparing saliency maps to fixations 

% presence of each feature
clc
clear
close all
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/Eye tracking database';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/saliency model/JuddSaliencyModel';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/saliency model';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/matlabPyrTools/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/SaliencyToolbox/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/voc-release5/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/voc-release5/features/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/FaceDetect/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/features/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/LabelMeToolbox-master/imagemanipulation/';
addpath '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/gbvs/gbvs';

% demo for one image
load model;

datafolder = "Eye tracking database/DATA/hp";
files=dir(fullfile(datafolder,'*.mat'));
[filenames{1:size(files,1)}] = deal(files.name);

subject_names = {"CNG", "ajs", "emb", "ems", "ff", "hp", "jcw", "jw", "kae", "krl", "po", "tmj", "tu", "ya", "zb"};

P = randperm(numel(filenames));

filenames = filenames(P(1:100));
scores_first_half = zeros(numel(subject_names), numel(filenames), 8);
scores_second_half = zeros(numel(subject_names), numel(filenames), 8);
scores_first_half_absence = zeros(numel(subject_names), numel(filenames), 8);
scores_second_half_absence = zeros(numel(subject_names), numel(filenames), 8);

file_no = 0;
for file_name = filenames
    file_no = file_no+1
    file_name = file_name{1};
    file_name = file_name(1:end-4);
    try
        datafolder = "Eye tracking database/";
        img_name = datafolder+"ALLSTIMULI/"+file_name+".jpeg";
        img = imread(img_name);
        FEATURES = saliency(img_name);

        % whiten the feature data with the parameters from the model.
        meanVec = model.whiteningParams(1, :);
        stdVec = model.whiteningParams(2, :);
        FEATURES=FEATURES-repmat(meanVec, [size(FEATURES, 1), 1]);
        FEATURES=FEATURES./repmat(stdVec, [size(FEATURES, 1), 1]);
    catch
        disp("ERROR - " + file_name)
        continue
    end

    [w_img, h, c] = size(img);
    dims = [200, 200];
    map{1} = img;

    origimgsize = size(img);
    origimgsize = origimgsize(1:2);

    subject_no = 0;
    
    for subject_name = subject_names

        subject_no = subject_no+1;

        data_file = datafolder+"DATA/"+subject_name+"/"+file_name+".mat";
        data = load(data_file);
        data = data.(sprintf("%s", file_name)).DATA.eyeData;

        X_first_half = data(1:floor(end/2), 1);
        Y_first_half = data(1:floor(end/2), 2);
        X_first_half = X_first_half(1:50);
        Y_first_half = Y_first_half(1:50);
        
        X_second_half = data(floor(end/2)+1:end, 1);
        Y_second_half = data(floor(end/2)+1:end, 2);
        X_second_half = X_second_half(1:50);
        Y_second_half = Y_second_half(1:50);

        for type = 1:8
            w = model.w;
            if (type==1) %subband
                w(:, 14:end) = 0;
            elseif (type==2) %Itti
                w(:, 1:13) = 0;
                w(:, 17:end) = 0;
            elseif (type==3) %Color
                w(1:16) = 0;
                w(:, 28:end) = 0;
            elseif type==4 %Torralba
                w(:, 1:27) = 0;
                w(:, 29:end) = 0;
            elseif type==5 %Horizon
                w(:, 1:28) = 0;
                w(:, 30:end) = 0;
            elseif type==6 %Object
                w(:, 1:29) = 0;
                w(:, 32:end) = 0;
            elseif type==7 %Center
                w(:, 1:32) = 0;
            elseif type==8 %all
                w = model.w;
            end
            
            % find the saliency map given the features and the model
            saliencyMap = (FEATURES*w(1:end-1)') + w(end);
            saliencyMap = (saliencyMap-min(saliencyMap))/(max(saliencyMap)-min(saliencyMap));
            saliencyMap = reshape(saliencyMap, dims);
            saliencyMap = imresize(saliencyMap, [w_img, h]);

            score_first_half = rocScoreSaliencyVsFixations(saliencyMap,X_first_half,Y_first_half,origimgsize);
            scores_first_half(subject_no, file_no, type) = score_first_half;

            score_second_half = rocScoreSaliencyVsFixations(saliencyMap,X_second_half,Y_second_half,origimgsize);
            scores_second_half(subject_no, file_no, type) = score_second_half;
        end
        
        for type = 1:8
            w = model.w;
            if (type==1) %subband
                w(:, 1:13) = 0;
            elseif (type==2) %Itti
                w(:, 14:16) = 0;
            elseif (type==3) %Color
                w(:, 17:27) = 0;
            elseif type==4 %Torralba
                w(:, 28) = 0;
            elseif type==5 %Horizon
                w(:, 29) = 0;
            elseif type==6 %Object
                w(:, 31) = 0;
            elseif type==7 %Center
                w(:, 33) = 0;
            elseif type==8 %all
                w = model.w;
            end
            
            % find the saliency map given the features and the model
            saliencyMap = (FEATURES*w(1:end-1)') + w(end);
            saliencyMap = (saliencyMap-min(saliencyMap))/(max(saliencyMap)-min(saliencyMap));
            saliencyMap = reshape(saliencyMap, dims);
            saliencyMap = imresize(saliencyMap, [w_img, h]);

            score_first_half = rocScoreSaliencyVsFixations(saliencyMap,X_first_half,Y_first_half,origimgsize);
            scores_first_half_absence(subject_no, file_no, type) = score_first_half;

            score_second_half = rocScoreSaliencyVsFixations(saliencyMap,X_second_half,Y_second_half,origimgsize);
            scores_second_half_absence(subject_no, file_no, type) = score_second_half;
        end
    end
end

save('Scores.mat', 'scores_first_half', 'scores_second_half')
save('Scores_absence.mat', 'scores_first_half_absence', 'scores_second_half_absence')
load Scores.mat
load Scores_absence.mat

for type = 1:8
    figure
    hist = histogram(scores_first_half(:,:,type), 'Normalization', 'pdf');
    values_first = hist.Values;
    edges_first = hist.BinEdges;
    edges_first = (edges_first(1:end-1)+edges_first(2:end))/2;
    hist = histogram(scores_second_half(:,:,type), 'Normalization', 'pdf');
    values_second = hist.Values;
    edges_second = hist.BinEdges;
    edges_second = (edges_second(1:end-1)+edges_second(2:end))/2; 
    plot(edges_first, values_first, 'k', 'LineWidth', 2)
    hold on
    plot(edges_second, values_second, 'm', 'LineWidth', 2)
    hold off
    xlim([0.0 1.1])
    xlabel('AUC')
    ylabel('Probability Density')
    legend('First Half', 'Second Half', 'Location', 'northwest')
    save_figure("Report/images/score/type_"+num2str(type))
end

for type = 1:8
    figure
    hist = histogram(scores_first_half_absence(:,:,type), 'Normalization', 'pdf');
    values_first = hist.Values;
    edges_first = hist.BinEdges;
    edges_first = (edges_first(1:end-1)+edges_first(2:end))/2;
    hist = histogram(scores_second_half_absence(:,:,type), 'Normalization', 'pdf');
    values_second = hist.Values;
    edges_second = hist.BinEdges;
    edges_second = (edges_second(1:end-1)+edges_second(2:end))/2; 
    plot(edges_first, values_first, 'k', 'LineWidth', 2)
    hold on
    plot(edges_second, values_second, 'm', 'LineWidth', 2)
    hold off
    xlim([0.0 1.1])
    xlabel('AUC')
    ylabel('Probability Density')
    legend('First Half', 'Second Half', 'Location', 'northwest')
    save_figure("Report/images/score/type_"+num2str(type)+"_absence")
end









