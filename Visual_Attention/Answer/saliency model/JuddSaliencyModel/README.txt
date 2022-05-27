 ----------------------------------------------------------------------
 Matlab tools for "Learning to Predict Where Humans Look" ICCV 2009
 Tilke Judd, Kristen Ehinger, Fredo Durand, Antonio Torralba
 
 Copyright (c) 2010 Tilke Judd
 Distributed under the MIT License
 See MITlicense.txt file in the distribution folder.
 
 Contact: Tilke Judd at <tjudd@csail.mit.edu>
 ----------------------------------------------------------------------


--------------
Contents
--------------

This code package includes the following files:

- saliency.m is the code that loads in an image and returns a saliency map.  It finds all the necessary features (find*.m files) and then loads the model.mat file to get the weights for combining all the features in a linear model.

- findTorralbaSaliency.m (which uses torralbaSaliency.m)
- findSubbandFeatures.m
- findObjectFeatures.m  (which uses the Felzenszwalb Detectors and Haarcascade xml file)
- findIttiFeatures.m
- findHorizonFeatures.m (which uses AntonioGaussian.m)
- findDistToCenterFeatures.m
- findColorFeatures.m (which uses colorHistogram.m)


------------------
Extra Installation
-------------------
To get the code for features to work, install the following tools:

1) Steerable pyramids code
http://www.cns.nyu.edu/~eero/steerpyr/
Used to get the subbands of the steerable pyramids.
This is also used to get the Torralba saliency model feature.  

2) Itti and Koch Saliency Toolbox 
http://www.saliencytoolbox.net/index.html
Used to get the channels of the Itti and Koch saliency model.

3) Felzenszwalb car and person detectors
http://people.cs.uchicago.edu/~pff/latent/
Used to find people and cars in images.
My code works with version 3, not the most recent version 4.  The easiest solution is to use version 3.  If you'd like to update my code to work with version 4 I believe the main change needs to be in my call in findObjectFeatures->detect() which needs to be changed to gdetect().  

4) Viola Jones Face detector
http://www.mathworks.com/matlabcentral/fileexchange/19912-open-cv-viola-jones-face-detection-in-matlab
Used to find faces in images.

5) LabelMe Toolbox
needed for the LMgist.m which is used for the horizon code
http://labelme.csail.mit.edu/LabelMeToolbox/index.html
Also, you need to uncomment line 114
img = imresizecrop(img, param.imageSize, 'bilinear');
and comment out line 115
img = imresize(img, param.imageSize, 'bilinear'); %jhhays

---------------
Getting Started
----------------

After installing all necessary extra toolboxes, open matlab and run
saliency('sampleImage.jpeg');


Send feedback, suggestions and questions to Tilke Judd at <tjudd@csail.mit.edu>