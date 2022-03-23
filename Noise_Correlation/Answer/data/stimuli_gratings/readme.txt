Contains drifiting sinusoidal gratings movies.
The last three digits in filename correspond to the
orientation angle.

Each .mat has a (num_pixels x num_pixels x num_images) matrix,
saved in uint8.  


———————
Code to play the movie:

load('M_grating000.mat');
num_images = size(M,3);
f = figure;
for iimage = 1:num_images
	imagesc(M(:,:,iimage));
	colormap(gray);
	pause(0.04);
end
———————

