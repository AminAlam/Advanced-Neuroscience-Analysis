clc
clear
close all
load("data/ArrayData.mat")
load("data/CleanTrials.mat")
fs = 200;

num_channels = numel(chan);
% removing bad trials
for ch_no = 1:num_channels
    chan(ch_no).lfp = chan(ch_no).lfp(:, Intersect_Clean_Trials);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LFP analysis
%% part a - finding the most dominant frequency oscillation
num_trials = size(chan(1).lfp, 2);
for ch_no = 1:num_channels
    lfp_data = chan(ch_no).lfp;
    for trial_no = 1:num_trials
        trial_data = lfp_data(:, trial_no);
    end
end