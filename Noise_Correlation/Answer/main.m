clc
clear
close all

for monk_no = 1:3
    data = load("data/spikes_gratings/S_monkey"+num2str(monk_no)+".mat");
    monk_data{monk_no} = data.S;
    data = load("data/spikes_gratings/data_monkey"+num2str(monk_no)+"_gratings.mat");
    monk_data_stuff{monk_no} = data.data;
    data = load("data/spikes_gratings/infos_monkey"+num2str(monk_no)+".mat");
    monk_elec_info{monk_no} = data.keepNeurons_mat_tmp;
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 1
% finding maximum activity
clc
close all
activity_max = zeros(3, 12);

for monk_no = 1:3
    data = monk_data{monk_no};
    neurons = 1:size(data(1).mean_FRs, 1);
    for grating_no = 1:12
        activity = data(grating_no).mean_FRs;
        activity = mean(activity, 2);
        [row, ~] = find(activity==max(activity));
        activity_max(monk_no, grating_no) = row(1);
    end
end
%% plotting psths
clc
close all

for grating_no = 1:12
    figure
    hold on
    
    monk_no = 1;
    data = monk_data{monk_no};
    elec_info1 = monk_elec_info{monk_no};
    max_activity_neuron = activity_max(monk_no, grating_no);
    activity = data(grating_no).mean_FRs/2e-2;
    color = [0, 0, 0];
    lineWidth = 2;
    plot(0:20:980, activity(max_activity_neuron, :), 'color', color, 'LineWidth', lineWidth)
    
    monk_no = 2;
    data = monk_data{monk_no};
    elec_info2 = monk_elec_info{monk_no};
    max_activity_neuron = activity_max(monk_no, grating_no);
    activity = data(grating_no).mean_FRs/2e-2;
    color = [1, 0, 0];
    lineWidth = 2;
    plot(0:20:980, activity(max_activity_neuron, :), 'color', color, 'LineWidth', lineWidth)
    
    monk_no = 3;
    data = monk_data{monk_no};
    elec_info3 = monk_elec_info{monk_no};
    max_activity_neuron = activity_max(monk_no, grating_no);
    activity = data(grating_no).mean_FRs/2e-2;
    color = [0, 0, 1];
    lineWidth = 2;
    plot(0:20:980, activity(max_activity_neuron, :), 'color', color, 'LineWidth', lineWidth)
    
    for monk_no = 1:3
        if monk_no == 1
            color = [0, 0, 0]+0.4;
        elseif monk_no == 2
            color = [1, 0.6, 0.6];
        elseif monk_no == 3
            color = [0.6, 0.6, 1];
        end
        lineWidth = 1;
        data = monk_data{monk_no};
        max_activity_neuron = activity_max(monk_no, grating_no);
        activity = data(grating_no).mean_FRs/2e-2;
        plot(0:20:980, mean(activity, 1), 'color', color, 'LineWidth', lineWidth)
    end
    
    legend("Neuron No."+num2str(elec_info1(activity_max(1, grating_no)))+" | Monk No.1", ...
        "Neuron No."+num2str(elec_info2(activity_max(2, grating_no)))+" | Monk No.2", ...
        "Neuron No."+num2str(elec_info3(activity_max(3, grating_no)))+" | Monk No.3", ...
        "Avg | Monk No.1", "Avg | Monk No.2", "Avg | Monk No.3")
    xlabel('Time (ms)')
    ylabel('Firing Rate (Hz)')
    title("PSTH Plot for orientation = "+num2str((grating_no-1)*30))
end

%% plotting tuning curve
clc
close all

for monk_no = 1:3
    neuron_indx = 1;
    figure
    hold on
    neurons = unique(activity_max(monk_no, :));
    neurons = neurons([1,2]);
    activity_mat = zeros(length(neurons), 12);
    neurons_info = monk_elec_info{monk_no};
    for neuron_no = neurons
        for grating_no = 1:12
            data = monk_data{monk_no};
            data = data(grating_no).mean_FRs;
            activity = data(neuron_no, :);
            activity = mean(activity, 2);
            activity_mat(neuron_indx, grating_no) = activity/2e-2;    
        end
        if neuron_indx == 1
            color = 'k';
        else 
            color = 'b';
        end
        plot(0:30:330, activity_mat(neuron_indx,:), color, 'LineWidth', 2)
        neuron_indx = neuron_indx+1;
    end
    
    activity_mat = zeros(1, 12);
    for grating_no = 1:12
        data = monk_data{monk_no};
        data = data(grating_no).mean_FRs;
        activity = mean(data, 'all');
        activity_mat(1, grating_no) = activity/2e-2;    
    end
    plot(0:30:330, activity_mat(1,:), '--m', 'LineWidth', 1)
    xlabel('Orientation')
    ylabel('Firing Rate (Hz)')
    legend("Neuron No."+num2str(neurons_info(neurons(1))), ...
        "Neuron No."+num2str(neurons_info(neurons(2))), ...
        "Avg")
    title("Tuning Curves for Monk No."+num2str(monk_no))
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 2
% finding preferred graiting for each neuron
clc
close all
for monk_no = 1:3
    data = monk_data{monk_no};
    channels = monk_data_stuff{monk_no}.CHANNELS;
    loc_map = monk_data_stuff{monk_no}.MAP;
    neurons_info = monk_elec_info{monk_no};
    neurons = 1:size(data(1).mean_FRs, 1);
    preferred_grating_mat = zeros(1, length(neurons));
    preferred_grating_mat_tmp = zeros(12, length(neurons));
    for neuron_no = neurons
        for grating_no = 1:12
            data_grating = data(grating_no).mean_FRs;
            activity = data_grating(neuron_no, :);
            activity = mean(activity, 2);
            preferred_grating_mat_tmp(grating_no, neuron_no) = activity; 
        end
        [row, ~] = find(preferred_grating_mat_tmp(:, neuron_no)==max(preferred_grating_mat_tmp(:, neuron_no)));
        preferred_grating_mat(neuron_no) = row(1);
    end
    grating_map = zeros(size(loc_map, 1), size(loc_map, 2))*nan;
    for i = 1:size(loc_map, 1)
        for j = 1:size(loc_map, 2)
            elec_no = loc_map(i, j);
            if ~isnan(elec_no)
                [row, ~] = find(channels(:,1)==elec_no);
                if ~isempty(row)
                    target_neuron = row(1);               
                    [row, ~] = find(neurons_info==target_neuron);
                    if ~isempty(row)
                        grating_map(i, j) = (preferred_grating_mat(row)-1)*30;
                    end
                end
            end
        end
    end
    preferred_grating{monk_no} = grating_map;
end
%% plotting preferred graiting for neurons
clc
close all
for monk_no = 1:3
    figure
    plot_mat = preferred_grating{monk_no};
    plt = imagesc(plot_mat);
    title("Similar Preferred Orientations | Monk"+num2str(monk_no))
    caxis([0, 330])
    colorbar
    set(plt,'AlphaData',~isnan(plot_mat))
    colormap jet
    
    figure
    plot_mat = preferred_grating{monk_no};
    plot_mat(find(isnan(plot_mat))) = -30;
    plt = pcolor(1:10, 1:10, plot_mat);
    title("Similar Preferred Orientations | Monk"+num2str(monk_no))
    caxis([0, 330])
    colorbar
    shading interp
    axis ij
%     set(plt,'AlphaData',~isnan(plot_mat))
    colormap jet
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 3
% finding dependence of r_sc on sitance

% calculating tuning curves
clc
close all
for monk_no = 1:3
    neuron_indx = 1;
    data = monk_data{monk_no};
    neurons_info = monk_elec_info{monk_no};
    neurons = 1:size(data(1).mean_FRs, 1);
    activity_mat = zeros(length(neurons), 12);

    for neuron_no = neurons
        for grating_no = 1:12
            data = monk_data{monk_no};
            data = data(grating_no).mean_FRs;
            activity = data(neuron_no, :);
            activity = mean(activity, 2);
            activity_mat(neuron_indx, grating_no) = activity/2e-2;    
        end
        neuron_indx = neuron_indx+1;
    end
    tuning_curves{monk_no} = activity_mat;
end

%% calculating r_s
for monk_no = 1:3
    tuning_curve_mat = tuning_curves{monk_no};
    r_s_mat = zeros(size(tuning_curve_mat, 1));     
    for neuron_1 = 1:size(tuning_curve_mat, 1)-1
        for neuron_2 = neuron_1+1:size(tuning_curve_mat, 1)
            tmp = corrcoef(tuning_curve_mat(neuron_1,:), tuning_curve_mat(neuron_2, :));
            r_s_mat(neuron_1, neuron_2) = tmp(1,2); 
        end
    end
    r_s{monk_no} = r_s_mat;
end
%% calculating r_sc
for monk_no = 1:3
    data =  monk_data{monk_no};
    r_s_mat = r_s{monk_no};
    r_sc_mat = zeros(size(r_s_mat, 1));
    for neuron_1 = 1:size(r_s_mat, 1)-1
        for neuron_2 = neuron_1+1:size(r_s_mat, 1)
            firing_rate_mat = zeros(12*200, 2);
            counter = 1;
            for orientation_no = 1:12
                for trial_no = 1:200
                    counts = data(orientation_no).trial(trial_no).counts;
                    firing_rate_mat(counter, 1) = sum(counts(neuron_1, :));
                    firing_rate_mat(counter, 2) = sum(counts(neuron_2, :));
                    counter = counter+1;
                end
            end
            firing_rate_mat = zscore(firing_rate_mat);
            tmp = corrcoef(firing_rate_mat);
            r_sc_mat(neuron_1, neuron_2) = tmp(1, 2);
        end
    end
    r_sc{monk_no} = r_sc_mat;
end
%% calculating distance between neurons
distance_2_units = 400e-6;
for monk_no = 1:3
    data =  monk_data{monk_no};
    neurons_info = monk_elec_info{monk_no};
    loc_map = monk_data_stuff{monk_no}.MAP;
    channels = monk_data_stuff{monk_no}.CHANNELS;
    distance_mat = zeros(size(data(1).trial(1).spikes, 1));
    for neuron_1 = 1:size(data(1).trial(1).spikes, 1)-1
        for neuron_2 = neuron_1+1:size(data(1).trial(1).spikes, 1)
            elec_1 = channels(neuron_1);
            elec_2 = channels(neuron_2);
            [row1, col1] = find(loc_map==elec_1);
            [row2, col2] = find(loc_map==elec_2);
            dx = distance_2_units*abs(row1-row2);
            dy = distance_2_units*abs(col1-col2);
            distance = sqrt(dx^2+dy^2);
            distance_mat(neuron_1, neuron_2) = distance;
        end
    end
    elec_distances{monk_no} = distance_mat;
end
%% plotting r_sc vs elec_distances



