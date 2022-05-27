function basicFreeView(datafolder)

%%%% Template program to record eye movements over discrete static imgs
%%%% note: psychtoolbox syntax is for PTB-3
%%%% written Barbara Hidalgo-Sotelo, 01/08
%%%% modified by Tilke Judd June 16, 2008 in order to add input parameters

warning('off','MATLAB:dispatcher:InexactMatch')
rand('state',sum(100*clock));
hidecursor;
Priority(0);

global backgroundcolor

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETUP, 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SETUP: Folders %%%
init %creates & saves the file "folderNames.mat"
load folderNames
% Overwrite stimfolder variable to be taken in as a parameter
stimfolder = datafolder;

%% SETUP: Input Variables %%%
% Dialog box settings:
prompt={'Subject initials:';'Experiment Name:';'# repetitions';};
def={'xxx';'ExpNameHere';'1'};
title='Input Variables';
lineNo=1; userinput=inputdlg(prompt,title,lineNo,def);
subName = char(userinput(1));                    %Subject initials
expename = char(userinput(2));                   %Experiment name
Nreps=str2num(char(userinput(3)));               %Number of times each img shown

%% SETUP: Experiment Parameters %%%
durFixation = 600;  durFixation = durFixation/1000;   %Duration of central fixation to cue trial onset [sec]
durFeedback = 750;  durFeedback = durFeedback/1000;   %Duration of Feedback [sec]

maxSearchTime = 3;      %Max amount of time to search [sec]
maxTimeOnFixCross = 6;  %Max time to expire on gray central fixation [sec]

KeyEscape = 27; %escape key

%% SETUP: Calibration Tool Parameters %%%
calThreshold = 1;      % threshold for calibration test
maxCalibrations = 5;   % max number of calib-test rounds
testFirst = 0;         % if 1, test first; otherwise re-cal first

load calibParams      % loads 3 param structs: alignReg, calibAndTest, midExpTest

allowedDeviation = 150;  %Enforce Initial Fixation around this "invisible box"
allowedDeviationRect = [0 0 allowedDeviation allowedDeviation];

okDeviation = 40;   %Req'd argument for DrawFixation
Xwidth = 8;         %Req'd argument for DrawFixation

Nrecals_midExp = 0; %Counter for calibration mid-experiment (maxTimeOnFixCross exceeded)
Nrecals_randChecks = 0;

nTrials_check = 50; %Number of trials before randCheck of calibrations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETUP, 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SETUP: STIMULI STUFF %%%

% %Practice Trials
% files=dir(fullfile(practicefolder,'*.jpg'));          %get names of practice images
% [practiceNames{1:size(files,1)}] = deal(files.name);  %names of imgs in cell array
% Npractice = size(practiceNames,2);

%Test Trials
files=dir(fullfile(stimfolder,'*.jpeg'))            %get names of STIMULI
[stimnames{1:size(files,1)}] = deal(files.name);
Nstimuli = size(stimnames,2);

%% Stimulus presentation order: (here, randomize stimuli within each blk)
for b=1:Nreps;
    order = randperm(Nstimuli);
    [shuff_stimnames{1:length(stimnames)}] = stimnames{order};
    % Save the stimnames order that is made
    eval(['save ''' saveDataFolder subName '_shuff_stimnames''  shuff_stimnames']);
    presentationOrder{b} = shuff_stimnames;
end


%% SETUP: PTB WINDOWS STUFF %%%

Screen('Preference', 'SkipSyncTests', 0);  %sets a PTB preference about timing tests (0 runs tests)
screenNumber = max(Screen('Screens'));     %find the secondary monitor, if exists, else use main monitor

% Open a double buffered fullscreen window.
[w, wRect]=Screen('OpenWindow',screenNumber);
[screenWidth screenHeight] = RectSize(wRect);

% Find the color values which correspond to white and black.
white=WhiteIndex(screenNumber);
black=BlackIndex(screenNumber);
gray=GrayIndex(screenNumber);
backgroundcolor = gray;

% Set background color and do initial flip to show blank screen:
Screen('FillRect', w, backgroundcolor);
Screen('Flip', w);

% Set Coordinates for possible fixation cross locations
[centerPoint(1) centerPoint(2)] = RectCenter(wRect);

% Specify Text Type & Size
messageFont = 'Palatino Linotype';
messageSize = 15;
Screen('TextFont', w, messageFont);
Screen('TextSize', w, messageSize);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETUP, 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set-Up DATA FILES %%%%%
% Creates a variable for each stimuli, labeled with SUBJECT INFO:
mkdir(saveDataFolder, subName);
for i=1:Nstimuli
%     imgName = stimnames{i}(1:5);
    varName = stimnames{i}(1:find(stimnames{i}=='.')-1)

    %Create and initialize data files
    eval([varName '.subName = subName;'])
    eval([varName '.when = date;'])
    eval([varName '.Nreps = Nreps;'])
    eval([varName '.imgName = stimnames{i};'])
    eval([varName '.DATA = [];'])
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEGIN EXPERIMENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ListenChar(2)
%% 1. INITIAL CALIBRATION ----------------------------------------------
openEyeTracker;

%Align & register targets once:
Screen('FillRect', w, white);
calibrationTool(alignReg, w, wRect);

%Calibrate and Test:
[passCal numCals calibInfo] = calibrateToThreshold(calAndTest, w, wRect, calThreshold, maxCalibrations, testFirst);

%Store Data from Initial Calibration:
CalData.initialCalibration.passCal = passCal;
CalData.initialCalibration.numCals = numCals;
CalData.initialCalibration.calibInfo = calibInfo;
Screen('FillRect', w, backgroundcolor);
Screen('Flip', w);

%% 2. EXPERIMENT  ------------------------------------------------------

% Start recording eye tracking data
bytes = sendEyeTracker(hex2dec('84'));
WaitSecs(.5);
disp('Beginning eyetracker communication');

% Label each file with Subject/Group Info:  0-11-subName-11-0
sendEyeTracker(0);
sendEyeTracker(11); for i = 1:length(subName); sendEyeTracker(subName(i)); end; sendEyeTracker(11);
sendEyeTracker(0);

% STEP 1) Show Instructions:
DrawFixation(w,wRect,centerPoint,okDeviation,Xwidth,white)
Waitsecs(1)

Screen('TextFont', w, messageFont); Screen('TextSize', w, messageSize);
DrawFormattedText(w, 'You will see a series of 1000 images.  Look closely at each image. \n After viewing the images, you will have a memory test: \n You will be asked to indicate whether or not you have seen an image before.  \n Press any key to begin.', 'center', 'center', [0 0 0]);
Screen('Flip', w);
WaitSecs(0.2); KbWait; while KbCheck; end;

% STEP 2) Practice Trials:
DrawFixation(w,wRect,centerPoint,okDeviation,Xwidth,white)
Waitsecs(1)

Screen('TextFont', w, messageFont); Screen('TextSize', w, messageSize);
DrawFormattedText(w, 'No test trials for right now. \n Press any key when you are ready to begin.', 'center', 'center', [0 0 0]);
Screen('Flip', w);
WaitSecs(0.2); KbWait; while KbCheck; end;

trial = 0;
checkCalEa15 = 0;
firstRowFlg = 1;

%Loop for BLOCKS
for b=1:Nreps
    stimOrder = presentationOrder{b};

    %Loop for TRIALS
    for t=1:Nstimuli
        trial = trial+1;
        checkCalEa15 = checkCalEa15+1;

        % Send Eyetracker Trial Marks
        sendEyeTracker(0);
        sendEyeTracker(100); sendEyeTracker(b); sendEyeTracker(100);
        sendEyeTracker(110); sendEyeTracker(t); sendEyeTracker(110);
        sendEyeTracker(0);

        % Read image into memory 
        imgName = stimOrder{t};
        img = imread(fullfile(stimfolder,imgName)); %Stimuli folders have been added to Matlab path
        imgTxt=Screen('MakeTexture', w, img);

        % Determine img size:
        stimSize = [size(img,1) size(img,2)];
        xoffset = (screenWidth-stimSize(2))/2; yoffset = (screenHeight-stimSize(1))/2;

        Screen('FillRect', w, backgroundcolor);
        Screen('Flip', w);
        WaitSecs(0.25)

        % (1) GRAY SCREEN FIXATION CROSS
        fixationRect = CenterRectOnPoint(allowedDeviationRect,centerPoint(1),centerPoint(2));
        DrawFixation(w,wRect,centerPoint,okDeviation,Xwidth,white)

        %Wait for Fixation from Eyetracker:
        sendEyeTracker(99);
        [timeOnFixation1,elapsedTime,FixEyeData] = Wait4BoxFixation(w,fixationRect,durFixation,maxTimeOnFixCross);
        sendEyeTracker(0);
        
        %<if maxTimeOnFixCross exceeded, jump to a Recalibrate screen>
        while timeOnFixation1>=maxTimeOnFixCross
            %Calibrate and Test:
            [passCal numCals calibInfo] = calibrateToThreshold(calAndTest, w, wRect, calThreshold, maxCalibrations, testFirst);

            Nrecals_midExp = Nrecals_midExp+1;
            %Store Data from Calibration
            eval(['CalData.midExp' num2str(Nrecals_midExp) 'Calibration.calibInfo = calibInfo;']);
            eval(['CalData.midExp' num2str(Nrecals_midExp) 'Calibration.block = b;']);
            eval(['CalData.midExp' num2str(Nrecals_midExp) 'Calibration.trial = t;']);

            Screen('FillRect', w, backgroundcolor);
            Screen('Flip', w);
            WaitSecs(0.5); %Blank screen

            DrawFixation(w,wRect,centerPoint,okDeviation,Xwidth,white)
            [timeOnFixation1,elapsedTime,FixEyeData] = Wait4BoxFixation(w,fixationRect,durFixation,maxTimeOnFixCross);
        end
        

        % (2) DISPLAY IMAGE:
        Screen('DrawTexture', w, imgTxt); %draw img only onto backbuffer in prep. for (4)
        Screen('Flip', w);

        %Record eye movements until trial ends (MAX TIME or KeyPress)
        sendEyeTracker(96);
        idx = 0; response = 1000;
        elapsedTime = 0; startTime=getsecs;  
        while (elapsedTime < maxSearchTime) %& (response==1000)
            %save eyetracking data:
            idx=idx+1;
            eyeData(idx,:) = GetEyePosition;
            %check whether a key was pressed...
            %{
            [keyIsDown,secs,keyCode]=KbCheck;
            if keyIsDown                           % if so, get RT, & break loop
                response=find(keyCode); response=response(1);
                RT=(getsecs-startTime);
                break
            else
            %}
                elapsedTime = getsecs - startTime; % if not, check whether maxDispTime has elapsed
            %end
        end
        sendEyeTracker(0);
        if elapsedTime >= maxSearchTime; RT = elapsedTime; end

        %Blank Screen
        Screen('FillRect', w, backgroundcolor);
        Screen('Flip', w);
        WaitSecs(0.25)

        %Rescale eye movement data into image-space (instead of screen-space)
        eyeData(:,1) = eyeData(:,1) - xoffset; eyeData(:,2) = eyeData(:,2) - yoffset;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% SAVE EYE MOVEMENT DATA:
        blk = b;  varName = imgName(1:find(imgName=='.')-1);
        eval([varName '.DATA(blk).block = blk;'])
        eval([varName '.DATA(blk).trial = t;'])
        eval([varName '.DATA(blk).eyeData = eyeData;'])

        eval(['save ' saveDataFolder subName '\' varName ' '...
            varName]);


% labels = [ 'ExpName'; 'Date'; 'OsName'; 'BlockNo'; 'TrialNo'; 'ImgName'; 'RT'; ];

        %%% SAVE BEHAVIORAL DATA:
        % (1)experiment name, (2)date, (3)SubName, (4)BLOCK, (5)TRIAL, (6)Img Name, (7)RT
        fids = fopen(fullfile(saveDataFolder,[subName '.data']),'a');
%         if firstRowFlg; fprintf(fids,labels); firstRowFlg=0; end
        fprintf(fids,'%s %s %s %d %d %s %4.4f \n',...
            expename,date,subName,blk,t,imgName,RT);
        fclose(fids) %Close subject file
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% TEST CALIBRATION EVERY SO OFTEN
        if checkCalEa15==nTrials_check
            %Give the status update
            Screen('TextFont', w, messageFont); Screen('TextSize', w, messageSize);
            tstring = strcat('You have completed:', num2str(t), '/', num2str(Nstimuli), ' trials \n Press any key to go on');
            DrawFormattedText(w, tstring, 'center', 'center', [0 0 0]);
            Screen('Flip', w);
            WaitSecs(0.2); KbWait; while KbCheck; end;
            
            %Calibrate and Test:
            [passCal numCals calibInfo] = calibrateToThreshold(midExpTest, w, wRect, calThreshold, maxCalibrations, 1);

            Nrecals_randChecks = Nrecals_randChecks+1;
            %Store Data from Calibration
            eval(['CalData.check' num2str(Nrecals_randChecks) 'Calibration.calibInfo = calibInfo;']);
            eval(['CalData.check' num2str(Nrecals_randChecks) 'Calibration.block = b;']);
            eval(['CalData.check' num2str(Nrecals_randChecks) 'Calibration.trial = t;']);

            Screen('FillRect', w, backgroundcolor);
            Screen('Flip', w);
            WaitSecs(0.5); %Blank screen

            checkCalEa15 = 0; %Reset check every-70 counter
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Close image window so it doesn't overload memory and crash
        Screen('Close', imgTxt);
        
    end % of Loop for TRIALS

end % of Loop for BLOCKS

%%% SAVE CALIBRATION DATA
eval(['save ''' saveDataFolder subName '_CalibrationData''  CalData']);

%% 3. ENDING STUFF: stop recording & close serial port, windows, textures
ListenChar;
sendEyeTracker(hex2dec('88'));
closeEyeTracker
Screen('CloseAll');
ShowCursor;

fprintf('This concludes the experiment \n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function DrawFixation(w,wRect,pickPoint,okDeviation,Xwidth,Kolor, fixationRect)

Screen('FillRect', w, Kolor, CenterRectOnPoint([okDeviation - (Xwidth/2), 0, okDeviation + (Xwidth/2), okDeviation*2], pickPoint(1),pickPoint(2)));
Screen('FillRect', w, Kolor, CenterRectOnPoint([0, okDeviation - (Xwidth/2), okDeviation*2, okDeviation + (Xwidth/2)], pickPoint(1),pickPoint(2)));
if nargin>6;
    Screen('FrameRect', w, Kolor, CenterRectOnPoint(fixationRect, pickPoint(1),pickPoint(2)), 3)
end
Screen('Flip', w);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [RT,elapsedTime,eyeData] = Wait4BoxFixation(w,checkFixRect,durFixation,maxTime)

global backgroundcolor

if nargin<4; maxTime = 9999; end

%Wait for durFixation in the box or max time to be reached
elapsedTime = 0; idx = 0; startTime=getsecs; initiateFixTime = startTime;
while (elapsedTime < durFixation) & ((getsecs-startTime) < maxTime)
    %save eyetracking data:
    idx=idx+1;
    eyeData(idx,:) = GetEyePosition;
    if IsInRect(eyeData(idx,1), eyeData(idx,2), checkFixRect)
        elapsedTime = GetSecs - initiateFixTime;
    else
        initiateFixTime = GetSecs;
    end
end
RT = getsecs - startTime; %When WHILE Loop stops, either fixDur or max time reached

%Blank screen
Screen('FillRect', w, backgroundcolor);
Screen('Flip', w);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

