% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 1
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
%% finding maximum activity
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
    activity = data(grating_no).mean_FRs;
    color = [0, 0, 0];
    lineWidth = 2;
    plot(0:20:980, activity(max_activity_neuron, :), 'color', color, 'LineWidth', lineWidth)
    
    monk_no = 2;
    data = monk_data{monk_no};
    elec_info2 = monk_elec_info{monk_no};
    max_activity_neuron = activity_max(monk_no, grating_no);
    activity = data(grating_no).mean_FRs;
    color = [1, 0, 0];
    lineWidth = 2;
    plot(0:20:980, activity(max_activity_neuron, :), 'color', color, 'LineWidth', lineWidth)
    
    monk_no = 3;
    data = monk_data{monk_no};
    elec_info3 = monk_elec_info{monk_no};
    max_activity_neuron = activity_max(monk_no, grating_no);
    activity = data(grating_no).mean_FRs;
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
        activity = data(grating_no).mean_FRs;
        plot(0:20:980, mean(activity, 1), 'color', color, 'LineWidth', lineWidth)
    end
    
    legend("Neuron No."+num2str(elec_info1(activity_max(1, grating_no)))+" | Monk No.1", ...
        "Neuron No."+num2str(elec_info2(activity_max(2, grating_no)))+" | Monk No.2", ...
        "Neuron No."+num2str(elec_info3(activity_max(3, grating_no)))+" | Monk No.3", ...
        "Avg | Monk No.1", "Avg | Monk No.2", "Avg | Monk No.3")
    
    title("PSTH Plot for grating = "+num2str((grating_no-1)*30))
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
    elec_info = monk_elec_info{monk_no};
    for neuron_no = neurons
        for grating_no = 1:12
            data = monk_data{monk_no};
            data = data(grating_no).mean_FRs;
            activity = data(neuron_no, :);
            activity = mean(activity, 2);
            activity_mat(neuron_indx, grating_no) = activity;    
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
        activity_mat(1, grating_no) = activity;    
    end
    plot(0:30:330, activity_mat(1,:), '--m', 'LineWidth', 1)
    
    legend("Neuron No."+num2str(elec_info(neurons(1))), ...
        "Neuron No."+num2str(elec_info(neurons(2))), ...
        "Avg")
    title("Tuning Curves for Monk No."+num2str(monk_no))
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 2
% finding preferred graiting for each neuron
clc
close all
for monk_no = 1:3
    data = monk_data{monk_no};
    elec_info = monk_elec_info{monk_no};
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
    loc_map = monk_data_stuff{monk_no}.MAP;
    grating_map = zeros(size(loc_map, 1), size(loc_map, 2))*nan;
    for i = 1:size(loc_map, 1)
        for j = 1:size(loc_map, 2)
            elec_no = loc_map(i, j);
            if ~isnan(elec_no)
                [row, ~] = find(elec_info==elec_no);
                if ~isempty(row)
                    grating_map(i, j) = (preferred_grating_mat(row)-1)*30;
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
end