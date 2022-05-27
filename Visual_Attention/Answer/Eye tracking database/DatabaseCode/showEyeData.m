function showEyeData(datafolder, stimfolder)

%%%% Show eye movement data collected with %"basicFreeView_v1.m" 
%%%% written Barbara Hidalgo-Sotelo, 01/08
%%%% updated by Tilke Judd 09/09

%Get names of stimuli files in Data Folder:
files=dir(fullfile(datafolder,'*.mat'));
[filenames{1:size(files,1)}] = deal(files.name);
Nstimuli = size(filenames,2)

%Go through each data file one at a time
for i=1:Nstimuli

    load(fullfile(datafolder,filenames{i}))

    stimFile = eval([filenames{i}(1:end-4)]);
    Nreps = size(stimFile.DATA,2); % Nreps is always 1 for this experiment

    imgName = stimFile.imgName;
    if exist(fullfile(stimfolder,imgName))
        img = imread(fullfile(stimfolder,imgName));
        i
    else
        output = 'image file not found'
    end
    
    for b=1:Nreps
        %Load eye data & analyze fixations:
        eyeData = stimFile.DATA(b).eyeData;
        [eyeData Fix Sac] = checkFixations(eyeData);
        fixs = find(eyeData(:,3)==0);
        
        %Show eye data with img:
        figure;
        imshow(img); hold on;
        plot(eyeData(:,1),eyeData(:,2),'r.','MarkerSize',14); %Plot all data points (red dots)
        plot(eyeData(fixs,1),eyeData(fixs,2),'y.','MarkerSize',14); %Plot all fixations (yellow dots)
        
        % Add numbers to indicate which Fixation is displayed
        % Note that DO NOT include the first fixation in the center of the
        % image which happens when a user first views every image.
        % This happens because we require users to look at a center
        % crosshair before moving onto the next image.
        appropFix = floor(Fix.medianXY(2:end, :));  % we start at fixation 2
        for j = 1:length(appropFix)
            text (appropFix(j, 1), appropFix(j, 2), ['{\color{black}\bf', num2str(j), '}'], 'FontSize', 16, 'BackgroundColor', [1, 1, 0]);
        end
        
        save_figure("Report/images/eye_tacking/img"+num2str(i))
        pause
        close
    end
end
