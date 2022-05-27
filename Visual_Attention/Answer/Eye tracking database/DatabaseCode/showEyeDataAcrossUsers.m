
function showEyeDataAcrossUsers(datafolder_all, stimfolder, numFix)

% Tilke Judd June 26, 2008
% ShowEyeDataForImage should show the eyetracking data for all users in
% 'users' on a specified image.

users = {'CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb'};

colors = cell(8, 1);
colors{1} = 'r'; colors{2} = 'g'; colors{3} = 'b'; colors{4} = '#EDB120'; 
colors{5} = 'm'; colors{6} = '#D95319'; colors{7} = '#A2142F'; colors{8} = 'k'; 
colors{9} = 'r'; colors{10} = 'g'; colors{11} = 'b'; colors{12} = '#EDB120'; 
colors{13} = 'm'; colors{14} = '#D95319'; colors{15} = '#A2142F'; colors{16} = 'k'; 

% Cycle through all images
files = dir(strcat(stimfolder, '/*.jpeg'));
for i = 1:length(files)
    filename = files(i).name;
    % Get image
    img = imread(fullfile(stimfolder, filename));
    figure;
    imshow(img); hold on;

    i
    
    for j = 1:length(users)
        user = users{j};

        % Get eyetracking data for this image
        datafolder = [datafolder_all '/' user];
        datafile = strcat(filename(1:end-4), 'mat');
        load(fullfile(datafolder, datafile));
        stimFile = eval([datafile(1:end-4)]);
        eyeData = stimFile.DATA(1).eyeData;
        [eyeData Fix Sac] = checkFixations(eyeData);
        s=find(eyeData(:, 3)==2, 1)+1; % to avoid the first fixation
        eyeData=eyeData(s:end, :);
        fixs = find(eyeData(:,3)==0);
        
        % Add numbers and initials to indicate which and whos fixation is displayed
        if (numFix <= length(Fix.medianXY))
            user;
            appropFix = floor(Fix.medianXY(2:numFix, :));
            
            for k = 1:length(appropFix)
                text (appropFix(k, 1), appropFix(k, 2), ['{\color{white}\bf', num2str(k), user, '}'], 'FontSize', 12, 'BackgroundColor', colors{k});
            end      
        end
    end
    save_figure("Report/images/eye_tacking/acrossSubjects_img"+num2str(i))
    pause
    close
end
