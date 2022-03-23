% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 1 - plotting tuning curves
clc
close all

monk1_data = load("data/spikes_gratings/S_monkey1.mat");
monk1_data = monk1_data.S;

%% finding maximum activity
neurons = 1:83;
activity_max = zeros(1, 12);

for neuron_no = neurons
    for grating_no = 1:12
        activity = monk1_data(grating_no).mean_FRs;
        activity = mean(activity, 2);
        [row, ~] = find(activity==max(activity));
        activity_max(grating_no) = row(1);
    end
end

%% plotting psths
clc
close all
neurons = 1:83;

for grating_no = 1:12
    figure
    activity = monk1_data(grating_no).mean_FRs;
    for neuron_no = neurons
        if neuron_no == activity_max(grating_no)
            color = 'k';
        else
            color = [0 0 0]+0.8;
        end
        plot(0:20:980, activity(neuron_no, :), 'color', color)
        hold on
    end
    
end
%% plotting tuning curve
neurons = [24, 25 52, 18];
activity_mat = zeros(length(neurons), 12);
neuron_indx = 1;
figure

for neuron_no = neurons
    for grating_no = 1:12
        activity = monk1_data(grating_no).mean_FRs;
        activity = mean(activity, 2);
        activity = activity(neuron_no);
        activity_mat(neuron_indx, grating_no) = activity;
    end
    plot(0:30:330, activity_mat(neuron_indx,:))
    hold on
    neuron_indx = neuron_indx+1;
end