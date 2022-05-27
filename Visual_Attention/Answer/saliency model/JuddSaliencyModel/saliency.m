function FEATURES = saliency(imagefile)
%
% saliencyMap = saliency(imagefile)
% This finds the saliency map of a given input image

% ----------------------------------------------------------------------
% Matlab tools for "Learning to Predict Where Humans Look" ICCV 2009
% Tilke Judd, Kristen Ehinger, Fredo Durand, Antonio Torralba
% 
% Copyright (c) 2010 Tilke Judd
% Distributed under the MIT License
% See MITlicense.txt file in the distribution folder.
% 
% Contact: Tilke Judd at <tjudd@csail.mit.edu>
% ----------------------------------------------------------------------

% load the image
img = imread(imagefile);
[w, h, c] = size(img);
dims = [200, 200];
FEATURES(:, 1:13) = findSubbandFeatures(img, dims);
FEATURES(:, 14:16) = findIttiFeatures(img, dims);
FEATURES(:, 17:27) = findColorFeatures(img, dims);
FEATURES(:, 28) = findTorralbaSaliency(img, dims);
FEATURES(:, 29) = findHorizonFeatures(img, dims);
FEATURES(:, 30:31) = findObjectFeatures(img, dims);
FEATURES(:, 33) = findDistToCenterFeatures(img, dims);
 
end
