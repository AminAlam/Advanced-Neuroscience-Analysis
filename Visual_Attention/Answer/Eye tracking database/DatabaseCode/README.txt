Images, Data and Code for the Eye tracking database related to the paper

Learning to predict where people look 
Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba
International Conference on Computer Vision (ICCV) 2009

Contact: <Tilke Judd> "tjudd@csail.mit.edu" with feedback or questions
Date written: September 2009, updated Nov 2009

IMAGES: http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip
EYE TRACKING DATA: http://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip
HUMAN GROUND TRUTH FIXATION MAPS: http://people.csail.mit.edu/tjudd/WherePeopleLook/HumanFixationMaps.zip
CODE: 
basicFreeView_v1.m
checkFixations.m
showEyeData.m
showEyeDataAcrossUsers.m

The eye tracking data contains data from 15 users on 1003 images.
The users are {'CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb'}


------------------------------------------------------------------------------------------------
Quickstart:
1) Download the images (ALLSTIMULI.zip) and the eye tracking data (DATA.zip) and the related matlab code (Code.zip).
2) Unzip all folders within a container folder (i.e EyetrackingDatabaseFolder)
3) In matlab, navigate to within the code folder (i.e. EyetrackingDatabaseFolder/DatabaseCode)
4) Run
>> showEyeData('../DATA/hp', '../ALLSTIMULI')
to see the eye fixations for user 'hp'.
------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------
More details:

WHAT IS IN THE DATA FOLDER?
In the eye tracking DATA folder, for each user there is	
1) a shuff_stimnames.mat file which shows the order in which the images were shown to the user.  Images were randomly shuffled at the beginning of each experiment so each user saw images in a different order. 
2) a CalibrationData.mat file which shows how well each user passed each calibration test.  For reference only.
3) a folder with .mat file for each of the 1003 images seen.
If you open one of the .mat files for an image in matlab, you will see information about the fixations.  For example, go to the directly with the eye tracking data in it and type the following in matlab:
>> load ../DATA/ajs/i05june05_static_street_boston_p1010764
>> i05june05_static_street_boston_p1010764.DATA.eyeData
this is the raw x and y pixel locations of the eye for every sample that the eye tracker took. 
Negative locations are blinks. 

To change the raw eye tracker data to more useful fixation and saccade data, use checkFixations.m.  Read through the top of that file for more information.  Here is an example:
    filename = 'i05june05_static_street_boston_p1010764.jpeg';
    datafolder = '../DATA/hp';
    datafile = strcat(filename(1:end-4), 'mat');
    load(fullfile(datafolder, datafile));
    stimFile = eval([datafile(1:end-4)]);
    eyeData = stimFile.DATA(1).eyeData;
    [eyeData Fix Sac] = checkFixations(eyeData);
    fixs = find(eyeData(:,3)==0); % these are the indices of the fixations in the eyeData for a given image and user


SEE FIXATION LOCATIONS ON IMAGES
In order to extract clusters of fixation locations and see the locations overlaid on the image, do the following:
>> showEyeData ([path to eye data for one user], [path to original images])
for example, 
>> showEyeData('../DATA/hp', '../ALLSTIMULI')
press the space bar to move to the visualization of the next image.

You can also run 
>> showEyeDataAcrossUsers([path to images], [number of fixations per person to show])
>> showEyeDataAcrossUsers('../ALLSTIMULI', 6)
This shows the eye tracking center of the eye tracking fixations for all users on a specific image.  The fixations are color coded per user and indicate the order of fixations.  Press space to continue to the next image.


CODE TO RUN THE ORIGINAL EXPERIMENT
We used the code basicFreeView_v1.m to run the experiment on the eyetracker.  You don't need to use it to view results.  It is just there for reference.  You need to download the code for the Psychtoolbox (http://psychtoolbox.org/wikka.php?wakka=HomePage) library to run it.

WHAT IS IN THE FIXATION MAPS FOLDER?
These are the human "ground truth" fixation maps for the given images.  They were created by convolving a gaussian over the top 6 fixation locations for all 15 users.

