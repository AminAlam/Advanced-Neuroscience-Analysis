clc
clear
close all
% loading data
load 'UnitsData.mat';

nbins = 64;
window_length = 9;
centers = linspace(-1.2, 2, nbins);
%% Step 1
%########################
% Calculate the PSTH for the units and plot the average 
% PSTH for each condition of the task
%########################
clc
close all

% PSTH for some of the units
neuron_indx = floor(rand(1)*numel(Unit));
figure
PSTHPlot(Unit, neuron_indx, window_length, nbins, centers)
%% average PSTH for each condition of the task
figure
hold on
counts_all_mean = [];
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            counts_all = [counts_all; counts];
        end
        
        counts_all = mean(counts_all,1);
        counts_all_mean = [counts_all_mean; counts_all];
        plot(centers, counts_all, 'LineWidth', 1)
    end
end
counts_all_mean = mean(counts_all_mean, 1);

plot(centers, counts_all_mean, 'k', 'LineWidth', 2)
xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(counts_all_mean)-10, max(counts_all_mean)+10])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')
title("Avg PSTH of All Units")


box = [0 0 0.3 0.3];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg')
%% Step 2
%########################
% Using GLM analysis to find out which units significantly encode
% the task conditions which includes reward expected value and cue location 
%########################
clc
close all

% Modelling left and rigth cue
neuron_indxs = 1:numel(Unit);
pVals = [];

for neuron_indx = neuron_indxs
    counts_all = [];
    trials_indx = [];

    pos = 1;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value(2) == pos
            trials_indx = [trials_indx;cnd.TrialIdx];
        end
    end

    y = [];
    all_counts = [];

    for i = 1:numel(Unit(neuron_indx).Trls)
        data = Unit(neuron_indx).Trls(i);
        [counts,~] = PSTH(data, window_length, nbins, centers);
        all_counts = [all_counts; counts];
        if ~isempty(find(trials_indx==i))
            y = [y; 1];
        else
            y = [y; 0];
        end
    end

    mdl = fitglm(all_counts, y);
    pVal = coefTest(mdl);
    pVals = [pVals, pVal];
end

[~, col] = find(pVals<0.01);
best_units_cueLR = neuron_indxs(col);

figure
plot_indx = 1;
for neuron_indx = best_units_cueLR(randi(length(best_units_cueLR), 15, 1))
    subplot(3, 5, plot_indx)
    PSTHPlot(Unit, neuron_indx, window_length, nbins, centers)
    legend('off')
    title("PSTH for Unit No "+num2str(neuron_indx)+" | Pvalue "+num2str(pVals(neuron_indx), 4))
    plot_indx = plot_indx+1;
end
legend({'[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg'})
LR_pVals_mean = mean(pVals);

figure
histogram(pVals, 'Normalization', 'pdf')
xlabel('pValue')
ylabel('counts')
title('Histogram of pValues for Cue Pos')


figure
hold on
counts_all_mean = [];
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        
        for neuron_indx = best_units_cueLR
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            counts_all = [counts_all; counts];
        end
        
        counts_all = mean(counts_all,1);
        counts_all_mean = [counts_all_mean; counts_all];
        plot(centers, counts_all, 'LineWidth', 1)
    end
end
counts_all_mean = mean(counts_all_mean, 1);

plot(centers, counts_all_mean, 'k', 'LineWidth', 2)
xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(counts_all_mean)-10, max(counts_all_mean)+10])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')


box = [0 0 0.3 0.3];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg')
%% Modelling Reward expected value
pVals = [];

for neuron_indx = neuron_indxs
    counts_all = [];
    trials_indx_1 = [];
    trials_indx_2 = [];
    trials_indx_3 = [];

    EV = 3;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value(1) == EV
            trials_indx_1 = [trials_indx_1; cnd.TrialIdx];
        end
    end
    
    EV = 6;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value(1) == EV
            trials_indx_2 = [trials_indx_2; cnd.TrialIdx];
        end
    end
    
    EV = 9;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value(1) == EV
            trials_indx_3 = [trials_indx_3; cnd.TrialIdx];
        end
    end

    y = [];
    all_counts = [];

    for i = 1:numel(Unit(neuron_indx).Trls)
        data = Unit(neuron_indx).Trls(i);
        [counts,~] = PSTH(data, window_length, nbins, centers);
        all_counts = [all_counts; counts];
        if ~isempty(find(trials_indx_1==i))
            y = [y; 3];
        elseif ~isempty(find(trials_indx_2==i))
            y = [y; 6];
        elseif ~isempty(find(trials_indx_3==i))
            y = [y; 9];
        end
    end

    mdl = fitglm(all_counts, y);
    pVal = coefTest(mdl);
    pVals = [pVals, pVal];
end

[~, col] = find(pVals<0.01);
best_units_EV = neuron_indxs(col);

figure
plot_indx = 1;

for neuron_indx = best_units_EV(randi(length(best_units_EV), 15, 1))
    subplot(3, 5, plot_indx)
    PSTHPlot(Unit, neuron_indx, window_length, nbins, centers)
    legend('off')
    title("PSTH for Unit No "+num2str(neuron_indx)+" | Pvalue "+num2str(pVals(neuron_indx), 4))
    plot_indx = plot_indx+1;
end

legend({'[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg'})
mean(pVals)
REV_pVals_mean = mean(pVals);

figure
histogram(pVals, 'Normalization', 'pdf')
xlabel('pValue')
ylabel('counts')
title('Histogram of pValues for Cue reward EV')


figure
hold on
counts_all_mean = [];
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        
        for neuron_indx = best_units_EV
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            counts_all = [counts_all; counts];
        end
        
        counts_all = mean(counts_all,1);
        counts_all_mean = [counts_all_mean; counts_all];
        plot(centers, counts_all, 'LineWidth', 1)
    end
end
counts_all_mean = mean(counts_all_mean, 1);

plot(centers, counts_all_mean, 'k', 'LineWidth', 2)
xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(counts_all_mean)-10, max(counts_all_mean)+10])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')


box = [0 0 0.3 0.3];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg')
%% Modelling left and rigth cue  and also Reward expected value

pVals = [];

for neuron_indx = neuron_indxs
    counts_all = [];
    trials_indx_1 = [];
    trials_indx_2 = [];
    trials_indx_3 = [];
    trials_indx_4 = [];
    trials_indx_5 = [];
    trials_indx_6 = [];
    
    pos = -1;
    
    EV = 3;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value == [EV, pos]
            trials_indx_1 = [trials_indx_1; cnd.TrialIdx];
        end
    end
    EV = 6;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value == [EV, pos]
            trials_indx_2 = [trials_indx_2; cnd.TrialIdx];
        end
    end
    EV = 9;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value == [EV, pos]
            trials_indx_3 = [trials_indx_3; cnd.TrialIdx];
        end
    end
    
    pos = 1;
    
    EV = 3;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value == [EV, pos]
            trials_indx_4 = [trials_indx_4; cnd.TrialIdx];
        end
    end
    EV = 6;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value == [EV, pos]
            trials_indx_5 = [trials_indx_5; cnd.TrialIdx];
        end
    end
    EV = 9;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value == [EV, pos]
            trials_indx_6 = [trials_indx_6; cnd.TrialIdx];
        end
    end

    y = [];
    all_counts = [];

    for i = 1:numel(Unit(neuron_indx).Trls)
        data = Unit(neuron_indx).Trls(i);
        [counts,~] = PSTH(data, window_length, nbins, centers);
        all_counts = [all_counts; counts];
        if ~isempty(find(trials_indx_1==i))
            y = [y; 1];
        elseif ~isempty(find(trials_indx_2==i))
            y = [y; 2];
        elseif ~isempty(find(trials_indx_3==i))
            y = [y; 3];
        elseif ~isempty(find(trials_indx_4==i))
            y = [y; 4];
        elseif ~isempty(find(trials_indx_5==i))
            y = [y; 5];
        elseif ~isempty(find(trials_indx_6==i))
            y = [y; 6];
        end
    end

    mdl = fitglm(all_counts, y);
    pVal = coefTest(mdl);
    pVals = [pVals, pVal];
end

[~, col] = find(pVals<0.01);
best_units = neuron_indxs(col);

figure
plot_indx = 1;

for neuron_indx = best_units(randi(length(best_units), 15, 1))
    subplot(3, 5, plot_indx)
    PSTHPlot(Unit, neuron_indx, window_length, nbins, centers)
    legend('off')
    title("PSTH for Unit No "+num2str(neuron_indx)+" | Pvalue "+num2str(pVals(neuron_indx), 4))
    plot_indx = plot_indx+1;
end

legend({'[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg'})
REV_LR_pVals_mean = mean(pVals);

figure
histogram(pVals, 'Normalization', 'pdf')
xlabel('pValue')
ylabel('counts')
title('Histogram of pValues for all 6 conditons')



figure
hold on
counts_all_mean = [];
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        
        for neuron_indx = best_units
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            counts_all = [counts_all; counts];
        end
        
        counts_all = mean(counts_all,1);
        counts_all_mean = [counts_all_mean; counts_all];
        plot(centers, counts_all, 'LineWidth', 1)
    end
end
counts_all_mean = mean(counts_all_mean, 1);

plot(centers, counts_all_mean, 'k', 'LineWidth', 2)
xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(counts_all_mean)-10, max(counts_all_mean)+10])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')


box = [0 0 0.3 0.3];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(counts_all_mean)-10 max(counts_all_mean)+10 max(counts_all_mean)+10 min(counts_all_mean)-10];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg')
%% Functions

function data = get_condition(Unit, neuron_indx, value)
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value == value
            trials_indx = cnd.TrialIdx;
            data = Unit(neuron_indx).Trls(trials_indx);
            break
        end
    end
end

function [counts,centers] = PSTH(data, window_length, nbins, centers)
    data_all = zeros(numel(data), nbins);
    for i=1:numel(data)
        [counts,centers] = hist(cell2mat(data(i)), nbins, 'xbins', centers);
        counts = movmean(counts, window_length);
        data_all(i, :) = counts;
    end
    data_all = mean(data_all,1);
    counts = data_all/(3.2/nbins);

end


function PSTHPlot(Unit, neuron_indx, window_length, nbins, centers)
    hold on
    counts_all = [];
    for cue_value = 3:3:9
        for pos = [-1, 1]
            value = [cue_value, pos];
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            counts_all = [counts_all; counts];
            plot(centers, counts, 'LineWidth', 1)
        end
    end
    counts_all = mean(counts_all,1);
    plot(centers, counts_all, 'k', 'LineWidth', 2)
    xline(0,'r', 'Reward Cue');
    xline(0.3,'m','Delay Period');
    xline(0.9,'b', 'Reaction');
    xlim([-1.2, 2])
    ylim([min(counts_all)-10, max(counts_all)+10])
    xlabel("Time (s)")
    ylabel('Firing Rate (Hz)')
    title("PSTH for Unit No "+num2str(neuron_indx))

    box = [0 0 0.3 0.3];
    boxy = [min(counts_all)-10 max(counts_all)+10 max(counts_all)+10 min(counts_all)-10];
    patch(box,boxy,'r','FaceAlpha',0.1)

    box = [0.3 0.3 0.9 0.9];
    boxy = [min(counts_all)-10 max(counts_all)+10 max(counts_all)+10 min(counts_all)-10];
    patch(box,boxy,'m','FaceAlpha',0.1)

    box = [0.9 0.9 2 2];
    boxy = [min(counts_all)-10 max(counts_all)+10 max(counts_all)+10 min(counts_all)-10];
    patch(box,boxy,'b','FaceAlpha',0.1)

    legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg')
    hold off
end

