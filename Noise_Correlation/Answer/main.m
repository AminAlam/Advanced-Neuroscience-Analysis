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
    firing_rate_mat = zeros(2400, size(r_s_mat, 1));
    for neuron_no = 1:size(r_s_mat, 1)
        counter = 1;
        firing_rate_mat_tmp = zeros(2400, 1);
        for orientation_no = 1:12
            for trial_no = 1:200
                counts = data(orientation_no).trial(trial_no).counts;
                firing_rate_mat_tmp(counter, 1) = sum(counts(neuron_no, :));
                counter = counter+1;
            end
            firing_rate_mat_tmp(counter-200:counter-1) = zscore(firing_rate_mat_tmp(counter-200:counter-1));
        end
        firing_rate_mat(:, neuron_no) = firing_rate_mat_tmp;
    end
    r_sc_mat = corrcoef(firing_rate_mat);
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
            elec_1 = channels(neurons_info(neuron_1));
            elec_2 = channels(neurons_info(neuron_2));
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
clc
close all
for monk_no = 1:3
    
    r_s_mat = r_s{monk_no};
    r_sc_mat = r_sc{monk_no};
    distance_mat = elec_distances{monk_no};
    r_sc_vec = [];
    r_s_vec = [];
    distace_vec = [];
    for neuron_1 = 1:size(r_s_mat, 1)-1
        for neuron_2 = neuron_1+1:size(r_s_mat, 1)
            r_s_vec = [r_s_vec, r_s_mat(neuron_1, neuron_2)];
            r_sc_vec = [r_sc_vec, r_sc_mat(neuron_1, neuron_2)];
            distace_vec = [distace_vec, distance_mat(neuron_1, neuron_2)];
        end
    end

    [distace_vec_sorted, I] = sort(distace_vec);
    r_s_vec_sorted = r_s_vec(I);
    r_sc_vec_sorted = r_sc_vec(I);

    plot_all = zeros(4, 8, 3);

    counter = 1;
    for distance_value = 0.25e-3:0.5e-3:4.25e-3
        [~, col] = find(distace_vec_sorted>=distance_value-0.26e-3 & distace_vec_sorted<distance_value+0.26e-3);
        distance_tmp = distace_vec_sorted(col);
        r_s_tmp = r_s_vec_sorted(col);
        r_sc_tmp = r_sc_vec_sorted(col);

        [~, col_2] = find(r_s_tmp>=0.5);
        r_sc_tmp2 = r_sc_tmp(col_2);
        distance_tmp2 = mean(distance_tmp(col_2));
        r_sc_mean = mean(r_sc_tmp2);
        r_sc_var = var(r_sc_tmp2);
        plot_all(1,counter,:) = [distance_tmp2, r_sc_mean, r_sc_var];

        [~, col_2] = find(r_s_tmp>=0 & r_s_tmp<0.5);
        r_sc_tmp2 = r_sc_tmp(col_2);
        distance_tmp2 = mean(distance_tmp(col_2));
        r_sc_mean = mean(r_sc_tmp2);
        r_sc_var = var(r_sc_tmp2);
        plot_all(2,counter,:) = [distance_tmp2, r_sc_mean, r_sc_var];

        [~, col_2] = find(r_s_tmp>=-0.5 & r_s_tmp<0);
        r_sc_tmp2 = r_sc_tmp(col_2);
        distance_tmp2 = mean(distance_tmp(col_2));
        r_sc_mean = mean(r_sc_tmp2);
        r_sc_var = var(r_sc_tmp2);
        plot_all(3,counter,:) = [distance_tmp2, r_sc_mean, r_sc_var];
        
        [~, col_2] = find(r_s_tmp<-0.5);
        r_sc_tmp2 = r_sc_tmp(col_2);
        distance_tmp2 = mean(distance_tmp(col_2));
        r_sc_mean = mean(r_sc_tmp2);
        r_sc_var = var(r_sc_tmp2);
        plot_all(4,counter,:) = [distance_tmp2, r_sc_mean, r_sc_var];

        counter = counter+1;
    end

    figure
    hold on
    for i = 1:4
        plot(1000*plot_all(i,:,1), plot_all(i,:,2), 'color', [0,0,0]+i/8, 'LineWidth', 5-i)
    end
    for i = 1:4
        errorbar(1000*plot_all(i,:,1), plot_all(i,:,2), plot_all(i,:,3), 'color', [0,0,0]+i/8)
    end
    legend('r_s > 0.5', 'r_s 0 to 0.5', 'r_s -0.5 to 0', 'r_s < -0.5','NumColumns',2)
    xlim([0, 4.5])
    title("Mokey No."+num2str(monk_no))
    xlabel("Distance between electrodes (mm)")
    ylabel("Spike count correlation (r_{sc})")
end
%% plotting r_sc vs r_s
clc
close all
for monk_no = 1:3
    
    r_s_mat = r_s{monk_no};
    r_sc_mat = r_sc{monk_no};
    distance_mat = elec_distances{monk_no};
    r_sc_vec = [];
    r_s_vec = [];
    distace_vec = [];
    for neuron_1 = 1:size(r_s_mat, 1)-1
        for neuron_2 = neuron_1+1:size(r_s_mat, 1)
            r_s_vec = [r_s_vec, r_s_mat(neuron_1, neuron_2)];
            r_sc_vec = [r_sc_vec, r_sc_mat(neuron_1, neuron_2)];
            distace_vec = [distace_vec, distance_mat(neuron_1, neuron_2)];
        end
    end

    [r_s_vec_sorted, I] = sort(r_s_vec);
    distace_vec_sorted = distace_vec(I);
    r_sc_vec_sorted = r_sc_vec(I);

    plot_all = zeros(4, 7, 3);

    counter = 1;
    for r_s_value = -0.75:0.25:0.75
        [~, col] = find(r_s_vec_sorted>=r_s_value-0.126 & r_s_vec_sorted<r_s_value+0.126);
        r_s_tmp = r_s_vec_sorted(col);
        distance_tmp = distace_vec_sorted(col);
        r_sc_tmp = r_sc_vec_sorted(col);

        [~, col_2] = find(distance_tmp>=3e-3);
        r_sc_tmp2 = r_sc_tmp(col_2);
        r_s_tmp2 = mean(r_s_tmp(col_2));
        distance_tmp2 = distance_tmp(col_2);
        r_sc_mean = mean(r_sc_tmp2);
        r_sc_var = var(r_sc_tmp2);
        plot_all(1,counter,:) = [r_s_tmp2, r_sc_mean, r_sc_var];
        
        [~, col_2] = find(distance_tmp>=2e-3 & distance_tmp<3e-3);
        r_sc_tmp2 = r_sc_tmp(col_2);
        r_s_tmp2 = mean(r_s_tmp(col_2));
        distance_tmp2 = distance_tmp(col_2);
        r_sc_mean = mean(r_sc_tmp2);
        r_sc_var = var(r_sc_tmp2);
        plot_all(2,counter,:) = [r_s_tmp2, r_sc_mean, r_sc_var];
        
        [~, col_2] = find(distance_tmp>=1e-3 & distance_tmp<2e-3);
        r_sc_tmp2 = r_sc_tmp(col_2);
        r_s_tmp2 = mean(r_s_tmp(col_2));
        distance_tmp2 = distance_tmp(col_2);
        r_sc_mean = mean(r_sc_tmp2);
        r_sc_var = var(r_sc_tmp2);
        plot_all(3,counter,:) = [r_s_tmp2, r_sc_mean, r_sc_var];
        
        [~, col_2] = find(distance_tmp>=0 & distance_tmp<1e-3);
        r_sc_tmp2 = r_sc_tmp(col_2);
        r_s_tmp2 = mean(r_s_tmp(col_2));
        distance_tmp2 = distance_tmp(col_2);
        r_sc_mean = mean(r_sc_tmp2);
        r_sc_var = var(r_sc_tmp2);
        plot_all(4,counter,:) = [r_s_tmp2, r_sc_mean, r_sc_var];
        counter = counter+1;
    end

    figure
    hold on
    for i = 1:4
        plot(plot_all(i,:,1), plot_all(i,:,2), 'color', [0,0,0]+i/8, 'LineWidth', i)
    end
    for i = 1:4
        errorbar(plot_all(i,:,1), plot_all(i,:,2), plot_all(i,:,3), 'color', [0,0,0]+i/8)
    end
    legend('Distance > 3mm', 'Distance 2 to 3', 'Distance 1 to 2', 'Distance 1 to 2','Location','southeast','NumColumns',1)
    xlim([-1, 1])
    title("Mokey No."+num2str(monk_no))
    xlabel("Orientation tuning similarity (r_s)")
    ylabel("Spike count correlation (r_{sc})")
end

%% plotting r_sc vs r_s vs elec_distances
clc
close all

bin_step_rs = 0.4;
bin_length_rs = bin_step_rs/2;
r_s_values = -1:bin_step_rs:1;

bin_step_distance = 0.4e-3;
bin_length_distance = bin_step_distance/2;
distance_values = 0:bin_step_distance:4.4e-3;

for monk_no = 1:3
    r_s_mat = r_s{monk_no};
    r_sc_mat = r_sc{monk_no};
    distance_mat = elec_distances{monk_no};
    r_sc_vec = [];
    r_s_vec = [];
    distace_vec = [];
    for neuron_1 = 1:size(r_s_mat, 1)-1
        for neuron_2 = neuron_1+1:size(r_s_mat, 1)
            r_s_vec = [r_s_vec, r_s_mat(neuron_1, neuron_2)];
            r_sc_vec = [r_sc_vec, r_sc_mat(neuron_1, neuron_2)];
            distace_vec = [distace_vec, distance_mat(neuron_1, neuron_2)];
        end
    end    
    
    C_data = zeros(length(r_s_values), length(distance_values));
    counter_distance = 1;
    values_all = [r_s_vec; distace_vec; r_sc_vec];
    for distance_value = distance_values
        counter_rs = 1;
        for r_s_value = r_s_values
            [~, col] = find(values_all(1, :)>r_s_value-bin_length_rs & values_all(1, :)<r_s_value+bin_length_rs ...
                & values_all(2, :)>distance_value-bin_length_distance & values_all(2, :)<distance_value+bin_length_distance);
            if ~isempty(col)
                C_data(counter_rs, counter_distance) = values_all(3, col(1));
            end
            counter_rs = counter_rs+1;
        end
        counter_distance = counter_distance+1;
    end
    figure
    pcolor(distance_values*1000, r_s_values, C_data)
    colorbar
    shading interp
    colormap jet
    title("Mokey No."+num2str(monk_no))
    xlabel("Distance between electrodes (mm)")
    ylabel("Orientation tuning similarity (r_s)")
end
