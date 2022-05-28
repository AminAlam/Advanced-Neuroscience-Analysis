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

imagefile = '/Users/mohammadaminalamalhod/Documents/University/University/NeuroSience_Ghazi_Advanced/HWs/Visual_Attention/Answer/Eye tracking database/ALLSTIMULI/i64011654.jpeg';
FEATURES_all = saliency(imagefile);

load model;

img = imread(imagefile);
[w, h, c] = size(img);
dims = [200, 200];
map{1} = img;

figure
imshow(img)

for type = 1:8
    if (type==1) %subband
        FEATURES = FEATURES_all;
        FEATURES(:, 14:end) = 0;
    elseif (type==2) %Itti
        FEATURES = FEATURES_all;
        FEATURES(:, 1:13) = 0;
        FEATURES(:, 17:end) = 0; 
    elseif (type==3) %Color
        FEATURES = FEATURES_all;
        FEATURES(:, 1:16) = 0;
        FEATURES(:, 28:end) = 0;
    elseif type==4 %Torralba
        FEATURES = FEATURES_all;
        FEATURES(:, 1:27) = 0;
        FEATURES(:, 29:end) = 0;
    elseif type==5 %Horizon
        FEATURES = FEATURES_all;
        FEATURES(:, 1:28) = 0;
        FEATURES(:, 30:end) = 0;
    elseif type==6 %Object
        FEATURES = FEATURES_all;
        FEATURES(:, 1:29) = 0;
        FEATURES(:, 32:end) = 0;
    elseif type==7 %Center
        FEATURES = FEATURES_all;
        FEATURES(:, 1:31) = 0;
    elseif type==8 %all
        FEATURES = FEATURES_all;
    end

    % whiten the feature data with the parameters from the model.
    meanVec = model.whiteningParams(1, :);
    stdVec = model.whiteningParams(2, :);
    FEATURES=FEATURES-repmat(meanVec, [size(FEATURES, 1), 1]);
    FEATURES=FEATURES./repmat(stdVec, [size(FEATURES, 1), 1]);

    % find the saliency map given the features and the model
    saliencyMap = (FEATURES*model.w(1:end-1)') + model.w(end);
    saliencyMap = (saliencyMap-min(saliencyMap))/(max(saliencyMap)-min(saliencyMap));
    saliencyMap = reshape(saliencyMap, dims);
    saliencyMap = imresize(saliencyMap, [w, h]);
    
    figure
    imshow(saliencyMap)
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% comparing saliency maps to fixations 
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
% filenames = filenames(1:100);
scores_first_half = zeros(numel(filenames), 8);
scores_second_half = zeros(numel(filenames), 8);
file_no = 0;

for file_name = filenames
    file_no = file_no+1;
    try
        file_name = file_name{1};
        file_name = file_name(1:end-4)

        datafolder = "Eye tracking database/";
        data_file = datafolder+"DATA/"+"hp/"+file_name+".mat";
        data = load(data_file);
        data = data.(sprintf("%s", file_name)).DATA.eyeData;

        img_name = datafolder+"ALLSTIMULI/"+file_name+".jpeg";
        fixation_map_name = datafolder+"ALLFIXATIONMAPS/"+file_name+"_fixMap"+".jpg";

        datafolder = "saliency model/SaliencyMaps/";
        saliency_map_name = datafolder+file_name+"SM"+".jpg";

        img = imread(img_name);
        FEATURES_all = saliency(img_name);
        fixation_map = imread(fixation_map_name);

        [w, h, c] = size(img);
        dims = [200, 200];
        map{1} = img;

        origimgsize = size(img);
        origimgsize = origimgsize(1:2);
        X_first_half = data(1:floor(end/2), 1);
        Y_first_half = data(1:floor(end/2), 2);

        X_second_half = data(floor(end/2)+1:end, 1);
        Y_second_half = data(floor(end/2)+1:end, 2);

        for type = 1:8
            if (type==1) %subband
                FEATURES = FEATURES_all;
                FEATURES(:, 1:13) = 0;
            elseif (type==2) %Itti
                FEATURES = FEATURES_all;
                FEATURES(:, 14:16) = 0;
            elseif (type==3) %Color
                FEATURES = FEATURES_all;
                FEATURES(:, 17:27) = 0;
            elseif type==4 %Torralba
                FEATURES = FEATURES_all;
                FEATURES(:, 28) = 0;
            elseif type==5 %Horizon
                FEATURES = FEATURES_all;
                FEATURES(:, 29) = 0;
            elseif type==6 %Object
                FEATURES = FEATURES_all;
                FEATURES(:, 30:31) = 0;
            elseif type==7 %Center
                FEATURES = FEATURES_all;
                FEATURES(:, 32) = 0;
            elseif type==8 %all
                FEATURES = FEATURES_all;
            end

            % whiten the feature data with the parameters from the model.
            meanVec = model.whiteningParams(1, :);
            stdVec = model.whiteningParams(2, :);
            FEATURES=FEATURES-repmat(meanVec, [size(FEATURES, 1), 1]);
            FEATURES=FEATURES./repmat(stdVec, [size(FEATURES, 1), 1]);

            % find the saliency map given the features and the model
            saliencyMap = (FEATURES*model.w(1:end-1)') + model.w(end);
            saliencyMap = (saliencyMap-min(saliencyMap))/(max(saliencyMap)-min(saliencyMap));
            saliencyMap = reshape(saliencyMap, dims);
            saliencyMap = imresize(saliencyMap, [w, h]);

            score_first_half = rocScoreSaliencyVsFixations(saliencyMap,X_first_half,Y_first_half,origimgsize);
            scores_first_half(file_no, type) = score_first_half;

            score_second_half = rocScoreSaliencyVsFixations(saliencyMap,X_second_half,Y_second_half,origimgsize);
            scores_second_half(file_no, type) = score_second_half;
        end
    catch
        disp('Err')
    end
end

%%
scores_first_half_no0 = scores_first_half;
scores_second_half_no0 = scores_second_half;

[row, ~] = find(scores_first_half_no0==0);
scores_first_half_no0(row(1), :) = [];

[row, ~] = find(scores_second_half_no0==0);
scores_second_half_no0(row(1), :) = [];

histogram(scores_first_half_no0, 'Normalization', 'pdf')
hold on
histogram(scores_second_half_no0, 'Normalization', 'pdf')

xlim([0 1])
