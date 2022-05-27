

%% Identify Saccades,Fixations
%% Barbara Hidalgo-Sotelo 12/18/05
%% This function takes some POR_X & POR_X [data(:,2)] and returns the data
%% with the data points labeled indicating whether a fixation or saccade
%% was occuring [data(:,3)]. The Fixation information (Nfixs,duration,medianXY)
%% and Saccade information (duration, distance [px, deg]) is returned [struct].


%%% OLD VERSION: function [data,Fix,Sac] = checkFixations(data,algo)
%%% Changes made to:
%%% add FixDur Cutoff, fix 'end-in-saccade' case, remove "algo" argument (set this in PARAMS)
%%% add 2 NaN flags (fieldnames in Fix & Sac)-
%%%     one for BAD eyetracking trial (NaN > 50% of trial: set in PARAMS, "badETdata")
%%%     one as optional argument to flag trials with X NaNs (can mark trials w/ a few NaNs easily)
%%% Barbara Hidalgo-Sotelo 6/28/06

function [data,Fix,Sac] = checkFixations(data,allowedNaNs)


%%%%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%
%% Acceleration-detection of Saccades:
DegPerSec = 6;                      % Acceleration threshold
pxPerDeg = 34;                      % based on current ET setup (display= 1024x768px = 40.5x30cm; distance=75cm)
accelCrit = pxPerDeg * DegPerSec;   %(34 px/deg)*(6 deg/sec) velocity diff. btwn t=0 & t=(stepSize/samplingRate)
winSize = 4;                        %4 samples = 16.667 ms
stepSize = 1;                       %1 samples =  4.167 ms
samplingRate = 240;

%% Minimum duration of fixation:
FixDurCut = 50;                   %Shortest fixation allowed [ms]
FixDurCut_pts = round(FixDurCut*samplingRate/1000); %convert above value to minumum # of data pts
% note- this excludes the duration of the final fixation (can be below this threshold)

%% NaN related params (checking data continuity)
badETdata = .5;         % Max percent of trial = NaN to set badEyeTrack flag
if nargin==2
   allowedNaNs = 6;    % If 2nd arg, this is the # of allowable NaNs for flag (0 or 1)
else allowedNaNs = [];  % Otherwise, this flag will be empty ( [] )
end

%% Which fixation algorithm to use:
algo = 'A';             % use Acceleration-based algorithm
% algo = 'N'; D=20;     % use NN-based algorithm (call other function)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Quality Control:
countNaN = length(find(isnan(data(:,1))==1));
percentNaN = countNaN/length(data);

% Flag 1- Check for BAD EyeTracking Data ("true" if data is >50% NaNs
badEyeTrack_flag =  percentNaN > badETdata;

% Flag 2- Check for 'acceptable' NaN levels ("true" if data has more than allowedNaNs)
if isempty(allowedNaNs)
   badNaN_flag = [];                        % only set flag if 2nd arg passed
else badNaN_flag = (countNaN > allowedNaNs);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FIND SACCADES & FIXATIONS:
switch algo
   case 'N'
       %% Option 1: Nearest Neighbors around D radius = fixation
       % ! last time i checked, this part didn't work, fix me ! (bhs 6/23/06)
       [fixation, Fix.medianXY(:,1), Fix.medianXY(:,2), Fix.start, Fix.end, fixdurations] = getFixationsAndTrajectories(data(:,1),data(:,2),D);
       saccades=find(fixation==0);    fixations=find(fixation==1);
       %If whole trial is at fixation
       if length(saccades) < 2
           Sac.duration = 1000*samplingRate*length(saccades);
           Fix.driftXY(1,1) = var(data(:,1));
           Fix.driftXY(1,2) = var(data(:,2));
       else %Else, find the jump btwn fixations...
           eachsac = cat(1,0,find(saccades(2:end)-saccades(1:end-1)~=1));
           eachsac = cat(1,eachsac,length(saccades));
           for s=2:length(eachsac)
               Sac.duration(s-1) = 1000*length(saccades(eachsac(s-1)+1:eachsac(s)))/240;
           end
           Sac.duration = 1000*Sac.duration/240';
       end
       Nfixs = length(fixdurations);
       Nsacs = length(Sac.duration);
       Fix.duration = 1000*samplingRate*fixdurations';
       Sac.distPx = (diff(Fix.medianXY(:,1)).^2 + diff(Fix.medianXY(:,2)).^2).^0.5;
       Sac.distDeg = Sac.distPx/pxPerDeg;
       Fix.badEyeTrack = badEyeTrack_flag; Fix.badNaN = badNaN_flag;
       Sac.badEyeTrack = badEyeTrack_flag; Sac.badNaN = badNaN_flag;
       return

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %% Option 2: Acceleration above a certain criteria = saccade
   case 'A'
       % Calculate distance btwn successive data pts (in px):
       posChange = (diff(data(:,1)).^2 + diff(data(:,2)).^2).^0.5; posChange = cat(1,0,posChange);
       % Smooth posChange with moving window average:
       posChangeSM = filter(ones(1,winSize)/winSize,1,posChange);

       % Find acceleration of 2 overlapping windows of duration (winSize/240), offset by (stepSize/240)
       velWin = zeros(length(posChange),2);
       for k = 2:length(posChange)-winSize-stepSize
           velWin(k,1) = sum(posChangeSM(k:k+winSize))/(winSize/samplingRate);
           velWin(k,2) = sum(posChangeSM(k+stepSize:k+winSize+stepSize))/(winSize/samplingRate);
       end
       accel = (velWin(:,2) - velWin(:,1));
       NOsac = 0;

       while NOsac==0  %Quit out of loop if no saccades detected (at various points)

           % Identify which samples exceeds the acceleration criterion
           checkSac = find(accel(:)>accelCrit);

           % If not more than 2 data pts that exceed accelCrit (QUIT):
           if length(checkSac) < 3;  break; end

           % Index checkSac with markers, 3-1-2 (beg-mid-end), for each saccade:
           eachsac1 =  find(checkSac(2:end)-checkSac(1:end-1) == 1); %successive data points = same saccade
           eachsac2 =  find(checkSac(2:end)-checkSac(1:end-1) > 1);  %just before checkSac jump =  end of saccade
           eachsac3 = find(eachsac2(2:end)-eachsac2(1:end-1) == 1);  %find "lone" pts (successive eachsac2's)
           eachsac3 = eachsac2(eachsac3+1);                            %(mark "lone" points w/ 3)
           if isempty(eachsac3);
               eachsac3 = [];
           end
           %Determine whether the last row in checkSac is...
           switch checkSac(end)-checkSac(end-1)
               case 1
                   eachsac1 = cat(1,eachsac1,length(checkSac)); % the end of a continuous saccade
               otherwise
                   eachsac3 = cat(1,eachsac3,length(checkSac)); % a 'lone' pt
           end
           n = zeros(length(checkSac),2); nn = [0 0]; n(:,1) = checkSac;
           % Continuous part of saccades, Mark by "1":
           n(eachsac1,2)=1;
           % End of (acceleration part) of saccades, Mark by "2":
           n(eachsac2,2)=2;
           % "Lone points", Mark by "3":
           n(eachsac3,2)=3;
           %Determine whether the last row in checkSac is...
           if checkSac(end)-checkSac(end-1)==1     %... end of a saccade
               n(end,2) = 2;
           else n = n(1:end-1,:);                  %... or a "lone" point (erase)
               eachsac2 = eachsac2(1:end-1);
           end
           % Beginning of saccade, Mark by ""3":
           eachsac3 = eachsac2+1; n(eachsac3,2)=3; %1+end = beg of saccade
           % Check the first data point in checkSac:
           if n(1,2)==1
               n(1,2)=3;
           end

           % Identify contiguous saccades, discard isolated data points above accelCrit
           idx=1;
           if (n(1,2)==3) & (n(2,2)==1)  %Enforce same rules as below
               nn(idx,:) = n(1,:); idx = idx+1;
           end
           for i=2:length(n)-1
               switch n(i,2)
                   case 3
                       if n(i+1,2)==1
                           nn(idx,:) = n(i,:);
                           idx=idx+1;
                       end
                   case 1
                       nn(idx,:) = n(i,:);
                       idx=idx+1;
                   case 2
                       if n(i-1,2)==1 & n(i+1,2)==3
                           nn(idx,:) = n(i,:);
                           idx=idx+1;
                       end
               end
               if i == length(n)-1      %Check last n(end,:)
                   if n(end,1)-nn(end,1) == 1
                       nn(idx,:) = n(i+1,:);
                   end
               end
           end
           checkSac=nn; clear nn n

           % If no saccades, Quit:
           if length(checkSac) < 3;  break; end

           % Use deceleration criteria to identify end of each saccade:
           eachsac2=find(checkSac(:,2)==2); eachsac3=find(checkSac(:,2)==3); %at this pt, 3-2 are pairs
           realSac = 0; badSac = []; goodSac = [];
           for sac = 1:length(eachsac2)
               check = [checkSac(eachsac2(sac))+1 checkSac(eachsac2(sac))+2];
               crosscheck = accel(check) < -accelCrit;
               stopafter = 25;  timeOK = 1;    %limit search for decelCrit-crossing for "stopafter" data pts
               while (((crosscheck(1,1)==1) & (crosscheck(2,1)==0))==0) & (timeOK==1)
                   check=check+1;
                   crosscheck = accel(check) < -accelCrit;
                   stopafter = stopafter-1;
                   timeOK = (max(check) < length(data)) & (stopafter>0); %Boolean variable to check for 1. end of data, 2. no deceleration
               end

               % 1) - no deceleration component found, thus not a saccade (EXCEPT...)
               if timeOK==0
                   %%% Unless Trial ends in saccade...
                   if checkSac(eachsac2) == find(accel~=0,1,'last')
                       goodSac = cat(1,goodSac,sac); % keep track of which 3-2 pairs are "good", real Saccades
                       realSac = realSac+1;
                       blah(realSac,1) = length(accel);
                   else
                       badSac = cat(1,badSac,sac);
                   end
               % 2) - deceleration found! (keep saccade)
               else
                   goodSac = cat(1,goodSac,sac); % keep track of which 3-2 pairs are "good", real Saccades
                   realSac = realSac+1;
                   blah(realSac,1) = check(1);    %end of saccade is just the data pt just PRIOR to crossing -accelCrit
               end
           end

           %Go through the GoodSac:
           if (isempty(goodSac)==0) & (isempty(checkSac)==0)
               goodSac2 = eachsac2(goodSac); goodSac3 = eachsac3(goodSac);
               decel = [checkSac(goodSac2) blah];

               %If there is more than 1 saccade, make sure that fixations meet cutoff-
               if realSac > 1
                   %This check excludes INITIAL fixation & a potential FINAL fixation:
                   tooShort = find(checkSac(goodSac3(2:end))-decel(1:end-1,2) <= FixDurCut_pts);
                   startSac = checkSac(goodSac3(setdiff(1:length(blah),tooShort+1)));
                   endSac = decel(setdiff(1:length(blah),tooShort),:);
               else
                   startSac = checkSac(goodSac3);
                   endSac = decel(end,:);
               end
               %%% Check FINAL fixation length here:
               if (size(data,1) - decel(end)) <= FixDurCut_pts
                   endSac(end) = size(data,1);
               end
           else
               %By eliminating all the badSac's, nothing left:
               startSac = []; endSac = [0 0]; endSac(1,:) = [];
           end

           % Label Saccades/Fixations, FINALLY:
           eachsac3=startSac; eachsac2 = endSac(:,2); eachsac1 = [];
           for f=1:length(eachsac3)
               eachsac1 = cat(2,eachsac1,(eachsac3(f)+1):(eachsac2(f)-1));
           end
           indexStuff = [(1:length(data))' zeros(1,length(data))'];
           indexStuff(eachsac3,2)=3; indexStuff(eachsac2,2)=2; indexStuff(eachsac1,2)=1;
           %%%

           % Tag eye movement data with fix/sac markers (add a third column):
           data(:,3) = indexStuff(:,2);

           % Find fixation Start/End saccade Start/End
           eachsac3=find(data(:,3)==3);  eachsac2=find(data(:,3)==2); %re-index saccade beg(3)/end(2)
           checkFixs = find(data(:,3)==0); checkSac = setdiff([1:length(data)],checkFixs)';

           % If no saccades, Quit:
           if length(checkSac) < 3;  break; end

           %If only 1 saccade:
           if length(eachsac3)==1
               % if 1 saccade, 2 fixations:
               if eachsac2(end) ~= length(data)
                   fixStart = cat(1,checkFixs(1),eachsac2+1);
                   % if 1 saccade, end in a saccade:
               else
                   fixStart = checkFixs(1);
               end
               %If >1 saccade:
           else
               fixStart = cat(1,checkFixs(1),eachsac2(1:end-1)+1);
               if eachsac2(end) ~= length(data)
                   fixStart(end+1) = eachsac2(end)+1;
               end
           end
           case1 = eachsac3(1)==1;                     %if they start in a saccade
           case2 = eachsac2(end) == length(data);      %if they end in a saccade
           if case1==0 & case2==0
               fixEnd = cat(1,eachsac3-1,length(data));
           elseif case1==1 & case2==0
               fixEnd = cat(1,eachsac3(2:end)-1,length(data));
           elseif case1==0 & case2==1
               fixEnd = eachsac3-1;
           elseif case1==1 & case2==1
               fixEnd = eachsac3(2:end)-1;
           end

           % Get fixation info median, duration, etc...
           Nfixs = length(fixStart);
           Nsacs = length(eachsac3);
           for i = 1:Nfixs
               Fix.duration(i)  = 1000*(fixEnd(i)-fixStart(i)+1)/samplingRate; % duration in seconds
               tmpX = data(fixStart(i):fixEnd(i),1);
               Fix.medianXY(i,1) = median(tmpX(~isnan(tmpX)));
               tmpY = data(fixStart(i):fixEnd(i),2);
               Fix.medianXY(i,2) = median(tmpY(~isnan(tmpY)));
               Fix.start(i) = fixStart(i);
               Fix.end(i)   = fixEnd(i);
               Fix.driftXY(i,1) = var(tmpX(~isnan(tmpX)));
               Fix.driftXY(i,2) = var(tmpY(~isnan(tmpY)));
           end
           Fix.badEyeTrack = badEyeTrack_flag; Fix.badNaN = badNaN_flag;

           Sac.duration = 1000*(eachsac2-eachsac3+1)/240; % duration in seconds
           Sac.distPx = (diff(Fix.medianXY(:,1)).^2 + diff(Fix.medianXY(:,2)).^2).^0.5;
           Sac.distDeg = Sac.distPx/pxPerDeg;
           Sac.badEyeTrack = badEyeTrack_flag; Sac.badNaN = badNaN_flag;
           return
       end

       %If this point is reached, is b/c no saccades...
       Nsacs=0; Nfixs = 1; %fixation duration = length(data);
       Fix.duration = 1000*(length(data))/samplingRate;
       Sac.duration = 0; Sac.distPx = 0; Sac.distDeg = 0;
       Fix.medianXY(1,1) = nanmedian(data(:,1));  Fix.medianXY(1,2) = nanmedian(data(:,2));
       Fix.start = 1; Fix.end = length(data);
       Fix.driftXY(1,1) = nanvar(data(:,1));
       Fix.driftXY(1,2) = nanvar(data(:,2));
       Fix.badEyeTrack = badEyeTrack_flag; Fix.badNaN = badNaN_flag;
       Sac.badEyeTrack = badEyeTrack_flag; Sac.badNaN = badNaN_flag;
       data(:,3) =  0;
end