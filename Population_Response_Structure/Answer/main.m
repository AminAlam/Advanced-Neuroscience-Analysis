clc
clear
close all
% loading data
load 'UnitsData.mat';

nbins = 64;
window_length = 9;
centers = linspace(-1.2, 2, nbins);
save_figures = 1;
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


if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step1_unit"+num2str(neuron_indx),'-dpng','-r0')
end
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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step1_AvgUnits",'-dpng','-r0')
end
%% Step 2
%########################
% Using GLM analysis to find out which units significantly encode
% the task conditions which includes reward expected value and cue location 
%########################
clc
close all

% Modelling left and rigth cue - Single Units
neuron_indxs = 1:numel(Unit);
pVals = [];

% shuffling boolian, 1 will shuffle the final labels
shuffle_bool = 1;

index_vec = 1:numel(Unit(1).Trls);
index_vec = index_vec(randperm(length(index_vec)));

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
    
    if shuffle_bool
        y = y(index_vec, :);
    end
    
    mdl = fitglm(all_counts, y);
    pVal = coefTest(mdl);
    pVals = [pVals, pVal];
end

[~, col] = find(pVals<0.01);
best_units_cueLR = neuron_indxs(col);

figure('units','normalized','outerposition',[0 0 1 1])
plot_indx = 1;
for neuron_indx = best_units_cueLR(randi(length(best_units_cueLR), 15, 1))
    subplot(3, 5, plot_indx)
    PSTHPlot(Unit, neuron_indx, window_length, nbins, centers)
    legend('off')
    title("PSTH for Unit No "+num2str(neuron_indx)+" | Pvalue "+num2str(pVals(neuron_indx), 4))
    plot_indx = plot_indx+1;
end
legend({'[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg'})
LR_pVals_mean = mean(pVals)

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_pos1_"+num2str(shuffle_bool),'-dpng','-r0')
end

figure
histogram(pVals, 'Normalization', 'pdf')
xlabel('pValue')
ylabel('Probability')
title('Histogram of pValues for Cue Pos (PDF Estimation)')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_pos2_"+num2str(shuffle_bool),'-dpng','-r0')
end

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
title('Avg PSTH of The Units with pValue < 0.01')


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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_pos3_"+num2str(shuffle_bool),'-dpng','-r0')
end
%% Modelling Reward expected value - Single Units
clc
close all
pVals = [];
neuron_indxs = 1:numel(Unit);

% shuffling boolian, 1 will shuffle the final labels
shuffle_bool = 1;

index_vec = 1:numel(Unit(1).Trls);
index_vec = index_vec(randperm(length(index_vec)));

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
    
    if shuffle_bool
        y = y(index_vec, :);
    end

    mdl = fitglm(all_counts, y);
    pVal = coefTest(mdl);
    pVals = [pVals, pVal];
end

[~, col] = find(pVals<0.01);
best_units_EV = neuron_indxs(col);

figure('units','normalized','outerposition',[0 0 1 1])
plot_indx = 1;
for neuron_indx = best_units_EV(randi(length(best_units_EV), 15, 1))
    subplot(3, 5, plot_indx)
    PSTHPlot(Unit, neuron_indx, window_length, nbins, centers)
    legend('off')
    title("PSTH for Unit No "+num2str(neuron_indx)+" | Pvalue "+num2str(pVals(neuron_indx), 4))
    plot_indx = plot_indx+1;
end

legend({'[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg'})
REV_pVals_mean = mean(pVals)

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_EV_pos1_"+num2str(shuffle_bool),'-dpng','-r0')
end


figure
histogram(pVals, 'Normalization', 'pdf')
xlabel('pValue')
ylabel('Probability')
title('Histogram of pValues for Reward Expected Value (PDF Estimation)')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_EV_pos2_"+num2str(shuffle_bool),'-dpng','-r0')
end

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
title('Avg PSTH of The Units with pValue < 0.01')

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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_EV_pos3_"+num2str(shuffle_bool),'-dpng','-r0')
end
%% Modelling left and rigth cue  and also Reward expected value - Single Units
clc
close all
pVals = [];
neuron_indxs = 1:numel(Unit);

% shuffling boolian, 1 will shuffle the final labels
shuffle_bool = 1;

index_vec = 1:numel(Unit(1).Trls);
index_vec = index_vec(randperm(length(index_vec)));

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
    
    if shuffle_bool
        y = y(index_vec);
    end
    
    mdl = fitglm(all_counts, y);
    pVal = coefTest(mdl);
    pVals = [pVals, pVal];
end

[~, col] = find(pVals<0.01);
best_units = neuron_indxs(col);

figure('units','normalized','outerposition',[0 0 1 1])
plot_indx = 1;
for neuron_indx = best_units(randi(length(best_units), 15, 1))
    subplot(3, 5, plot_indx)
    PSTHPlot(Unit, neuron_indx, window_length, nbins, centers)
    legend('off')
    title("PSTH for Unit No "+num2str(neuron_indx)+" | Pvalue "+num2str(pVals(neuron_indx), 4))
    plot_indx = plot_indx+1;
end

legend({'[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Avg'})
REV_allCond_pVals_mean = mean(pVals)

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_all_pos1_"+num2str(shuffle_bool),'-dpng','-r0')
end

figure
histogram(pVals, 'Normalization', 'pdf')
xlabel('pValue')
ylabel('Probability')
title('Histogram of pValues for All 6 Conditions (PDF Estimation)')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_all_pos2_"+num2str(shuffle_bool),'-dpng','-r0')
end

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
title('Avg PSTH of The Units with pValue < 0.01')

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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("step2_all_pos3_"+num2str(shuffle_bool),'-dpng','-r0')
end
%% Modelling left and rigth cue - Population
clc
close all

neuron_indxs = 1:numel(Unit);
y = [];
all_counts = [];

% shuffling boolian, 1 will shuffle the final labels
shuffle_bool = 1;
index_vec = 1:numel(Unit)*2;
index_vec = index_vec(randperm(length(index_vec)));

for neuron_indx = neuron_indxs
    counts_all = [];
    trials_indx_1 = [];
    trials_indx_2 = [];
    
    pos = -1;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value(2) == pos
            trials_indx_1 = [trials_indx_1; cnd.TrialIdx];
        end
    end
    pos = 1;
    for i = 1:numel(Unit(neuron_indx).Cnd)
        cnd = Unit(neuron_indx).Cnd(i);
        if cnd.Value(2) == pos
            trials_indx_2 = [trials_indx_2; cnd.TrialIdx];
        end
    end
    
    data = Unit(neuron_indx).Trls(trials_indx_1);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 0];

    data = Unit(neuron_indx).Trls(trials_indx_2);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 1];
end

if shuffle_bool
        y = y(index_vec);
end

mdl = fitglm(all_counts, y);
pVal_lr_population = coefTest(mdl)
%% Modelling Reward expected value - Population
clc
close all

neuron_indxs = 1:numel(Unit);
y = [];
all_counts = [];

% shuffling boolian, 1 will shuffle the final labels
shuffle_bool = 1;
index_vec = 1:numel(Unit)*3;
index_vec = index_vec(randperm(length(index_vec)));

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
    
    data = Unit(neuron_indx).Trls(trials_indx_1);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 3];

    data = Unit(neuron_indx).Trls(trials_indx_2);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 6];

    data = Unit(neuron_indx).Trls(trials_indx_3);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 9];
end

if shuffle_bool
        y = y(index_vec);
end

mdl = fitglm(all_counts, y);
pVal_EV_population = coefTest(mdl)
%% Modelling left and rigth cue and also Reward expected value - Population
clc
close all

neuron_indxs = 1:numel(Unit);
y = [];
all_counts = [];

% shuffling boolian, 1 will shuffle the final labels
shuffle_bool = 1;
index_vec = 1:numel(Unit)*6;
index_vec = index_vec(randperm(length(index_vec)));

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

    data = Unit(neuron_indx).Trls(trials_indx_1);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 1];

    data = Unit(neuron_indx).Trls(trials_indx_2);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 2];

    data = Unit(neuron_indx).Trls(trials_indx_3);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 3];

    data = Unit(neuron_indx).Trls(trials_indx_4);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 4];

    data = Unit(neuron_indx).Trls(trials_indx_5);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 5];

    data = Unit(neuron_indx).Trls(trials_indx_6);
    [counts,~] = PSTH(data, window_length, nbins, centers);
    all_counts = [all_counts; counts];
    y = [y; 6];
end

if shuffle_bool
        y = y(index_vec);
end

mdl = fitglm(all_counts, y);
pVal_LR_EV_population = coefTest(mdl)
%% Step 3
%########################
% Plotting the population activity in lower dimension by using 
% suitable dimension reduction algorithm.
%########################

% I have used PCA to reduce to dimensions
clc
close all

all_data = zeros(numel(Unit), nbins, 6);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
end

% reducing units dimensinos to 1 - all conditions
figure
hold on
all_data_reduced = zeros(size(all_data, 2), size(all_data, 3));
for i = 1:6
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,2:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,i) = Z(1, :);
    plot(centers, all_data_reduced(:, i), 'LineWidth', 2)
end

xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(all_data_reduced,[],'all')-1, max(all_data_reduced,[],'all')+1])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')
title("PSTH of Dimension-Reduced Activity")

box = [0 0 0.3 0.3];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]')
hold off
%% reducing units dimensinos to 2 - all conditions
clc
close all
figure
all_data_reduced = zeros(2, size(all_data, 2), size(all_data, 3));
for i = 1:6
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,3:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,:,i) = Z(1:2, :);
    plot3(centers, all_data_reduced(1,:,i), all_data_reduced(2,:,i), 'LineWidth', 2)
    hold on
end
xlabel('Time (s)')
ylabel('Dim 1 Units')
zlabel('Dim 2 Units')

xlim([-1.2, 2])


[Y,Z] = meshgrid(min(all_data_reduced(1,:,:),[],'all'):0.1:max(all_data_reduced(1,:,:),[],'all'),min(all_data_reduced(2,:,:),[],'all'):0.1:max(all_data_reduced(2,:,:),[],'all'));
X = Z*0;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'r');
s.EdgeColor = 'none';

X = Z*0+0.3;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'm');
s.EdgeColor = 'none';

X = Z*0+0.9;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'b');
s.EdgeColor = 'none';

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Reward Cue', 'Delay Period', 'Reaction')
hold off
%% Plotting First Three PCs of the data in 3D coords - all conditions
clc
close all

figure
all_data_reduced = zeros(3, size(all_data, 2), size(all_data, 3));
for i = 1:6
    cov_mat = cov(all_data(:,:,i));
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);

    all_data_reduced(:,:,i) = V(:, 1:3)';
    plot3(all_data_reduced(1,:,i), all_data_reduced(2,:,i), all_data_reduced(3,:,i), 'LineWidth', 2)
    hold on
end
xlabel('Dim 1 Units')
ylabel('Dim 2 Units')
zlabel('Dim 3 Units')

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]')
%% reducing units dimensinos to 1 - Just Expected Value
clc
close all

all_data_tmp = zeros(numel(Unit), nbins, 6);
all_data = zeros(2*numel(Unit), nbins, 3);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data_tmp(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
    all_data(:,:,cue_value/3) = [all_data_tmp(:,:,cue_indx-2); all_data_tmp(:,:,cue_indx-1)];
end

figure
hold on
all_data_reduced = zeros(size(all_data, 2), size(all_data, 3));
for i = 1:3
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,2:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,i) = Z(1, :);
    plot(centers, all_data_reduced(:, i), 'LineWidth', 2)
end

xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(all_data_reduced,[],'all')-1, max(all_data_reduced,[],'all')+1])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')
title("PSTH of Dimension-Reduced Activity")

box = [0 0 0.3 0.3];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('3', '6', '9', 'Reward Cue', 'Delay Period', 'Reaction')
hold off
%% reducing units dimensinos to 2 - Just Expected Value
clc
close all

all_data_tmp = zeros(numel(Unit), nbins, 6);
all_data = zeros(2*numel(Unit), nbins, 3);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data_tmp(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
    all_data(:,:,cue_value/3) = [all_data_tmp(:,:,cue_indx-2); all_data_tmp(:,:,cue_indx-1)];
end

figure
all_data_reduced = zeros(2, size(all_data, 2), size(all_data, 3));
for i = 1:3
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,3:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,:,i) = Z(1:2, :);
    plot3(centers, all_data_reduced(1,:,i), all_data_reduced(2,:,i), 'LineWidth', 2)
    hold on
end
xlabel('Time (s)')
ylabel('Dim 1 Units')
zlabel('Dim 2 Units')

xlim([-1.2, 2])


[Y,Z] = meshgrid(min(all_data_reduced(1,:,:),[],'all'):0.1:max(all_data_reduced(1,:,:),[],'all'),min(all_data_reduced(2,:,:),[],'all'):0.1:max(all_data_reduced(2,:,:),[],'all'));
X = Z*0;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'r');
s.EdgeColor = 'none';

X = Z*0+0.3;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'm');
s.EdgeColor = 'none';

X = Z*0+0.9;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'b');
s.EdgeColor = 'none';

legend('3', '6', '9', 'Reward Cue', 'Delay Period', 'Reaction')
hold off
%% Plotting First Three PCs of the data in 3D coords - Just Expected Value
clc
close all

all_data_tmp = zeros(numel(Unit), nbins, 6);
all_data = zeros(2*numel(Unit), nbins, 3);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data_tmp(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
    all_data(:,:,cue_value/3) = [all_data_tmp(:,:,cue_indx-2); all_data_tmp(:,:,cue_indx-1)];
end

figure
all_data_reduced = zeros(3, size(all_data, 2), size(all_data, 3));
for i = 1:3
    cov_mat = cov(all_data(:,:,i));
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);

    all_data_reduced(:,:,i) = V(:, 1:3)';
    plot3(all_data_reduced(1,:,i), all_data_reduced(2,:,i), all_data_reduced(3,:,i), 'LineWidth', 2)
    hold on
end
xlabel('Dim 1 Units')
ylabel('Dim 2 Units')
zlabel('Dim 3 Units')

legend('3', '6', '9')
%% Step 4
%########################
% Shuffling the data and repeating the above steps
%########################

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% normal shuffling

% reducing units dimensinos to 1 - all conditions
clc
% close all

% shuffling boolian, 1 will shuffle the Units labels
shuffle_bool = 0;

all_data = zeros(numel(Unit), nbins, 6);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
end

if shuffle_bool
    for neuron_indx = 1:numel(Unit)
        index_vec = 1:6;
        index_vec = index_vec(randperm(length(index_vec)));
        all_data(neuron_indx,:,:) = all_data(neuron_indx, :, index_vec);
    end
end

figure
hold on
all_data_reduced = zeros(size(all_data, 2), size(all_data, 3));

for i = 1:6
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,2:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,i) = Z(1, :);
    plot(centers, all_data_reduced(:, i), 'LineWidth', 2)
end

xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(all_data_reduced,[],'all')-1, max(all_data_reduced,[],'all')+1])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')
title("PSTH of Dimension-Reduced Activity")

box = [0 0 0.3 0.3];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]')
hold off
%% reducing units dimensinos to 2 - all conditions
clc
% close all

% shuffling boolian, 1 will shuffle the Units labels
shuffle_bool = 1;

all_data = zeros(numel(Unit), nbins, 6);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
end

if shuffle_bool
    for neuron_indx = 1:numel(Unit)
        index_vec = 1:6;
        index_vec = index_vec(randperm(length(index_vec)));
        all_data(neuron_indx,:,:) = all_data(neuron_indx, :, index_vec);
    end
end

figure
all_data_reduced = zeros(2, size(all_data, 2), size(all_data, 3));
for i = 1:6
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,3:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,:,i) = Z(1:2, :);
    plot3(centers, all_data_reduced(1,:,i), all_data_reduced(2,:,i), 'LineWidth', 2)
    hold on
end
xlabel('Time (s)')
ylabel('Dim 1 Units')
zlabel('Dim 2 Units')

xlim([-1.2, 2])


[Y,Z] = meshgrid(min(all_data_reduced(1,:,:),[],'all'):0.1:max(all_data_reduced(1,:,:),[],'all'),min(all_data_reduced(2,:,:),[],'all'):0.1:max(all_data_reduced(2,:,:),[],'all'));
X = Z*0;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'r');
s.EdgeColor = 'none';

X = Z*0+0.3;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'm');
s.EdgeColor = 'none';

X = Z*0+0.9;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'b');
s.EdgeColor = 'none';

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]', 'Reward Cue', 'Delay Period', 'Reaction')
hold off
%% Plotting First Three PCs of the data in 3D coords - all conditions
clc
% close all

% shuffling boolian, 1 will shuffle the Units labels
shuffle_bool = 0;

all_data = zeros(numel(Unit), nbins, 6);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
end

if shuffle_bool
    for neuron_indx = 1:numel(Unit)
        index_vec = 1:6;
        index_vec = index_vec(randperm(length(index_vec)));
        all_data(neuron_indx,:,:) = all_data(neuron_indx, :, index_vec);
    end
end

figure
all_data_reduced = zeros(3, size(all_data, 2), size(all_data, 3));
for i = 1:6
    cov_mat = cov(all_data(:,:,i));
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);

    all_data_reduced(:,:,i) = V(:, 1:3)';
    plot3(all_data_reduced(1,:,i), all_data_reduced(2,:,i), all_data_reduced(3,:,i), 'LineWidth', 2)
    hold on
end
xlabel('Dim 1 Units')
ylabel('Dim 2 Units')
zlabel('Dim 3 Units')

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]')
%% reducing units dimensinos to 1 - Just Expected Value
clc
% close all

% shuffling boolian, 1 will shuffle the Units labels
shuffle_bool = 1;

all_data_tmp = zeros(numel(Unit), nbins, 6);
all_data = zeros(2*numel(Unit), nbins, 3);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data_tmp(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
    all_data(:,:,cue_value/3) = [all_data_tmp(:,:,cue_indx-2); all_data_tmp(:,:,cue_indx-1)];
end

if shuffle_bool
    for neuron_indx = 1:numel(Unit)
        index_vec = 1:3;
        index_vec = index_vec(randperm(length(index_vec)));
        all_data(neuron_indx,:,:) = all_data(neuron_indx, :, index_vec);
    end
end

figure
hold on
all_data_reduced = zeros(size(all_data, 2), size(all_data, 3));
for i = 1:3
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,2:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,i) = Z(1, :);
    plot(centers, all_data_reduced(:, i), 'LineWidth', 2)
end

xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(all_data_reduced,[],'all')-1, max(all_data_reduced,[],'all')+1])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')
title("PSTH of Dimension-Reduced Activity")

box = [0 0 0.3 0.3];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('3', '6', '9', 'Reward Cue', 'Delay Period', 'Reaction')
hold off
%% reducing units dimensinos to 2 - Just Expected Value
clc
% close all

% shuffling boolian, 1 will shuffle the Units labels
shuffle_bool = 1;

all_data_tmp = zeros(numel(Unit), nbins, 6);
all_data = zeros(2*numel(Unit), nbins, 3);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data_tmp(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
    all_data(:,:,cue_value/3) = [all_data_tmp(:,:,cue_indx-2); all_data_tmp(:,:,cue_indx-1)];
end

if shuffle_bool
    for neuron_indx = 1:numel(Unit)
        index_vec = 1:3;
        index_vec = index_vec(randperm(length(index_vec)));
        all_data(neuron_indx,:,:) = all_data(neuron_indx, :, index_vec);
    end
end

figure
all_data_reduced = zeros(2, size(all_data, 2), size(all_data, 3));
for i = 1:3
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,3:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,:,i) = Z(1:2, :);
    plot3(centers, all_data_reduced(1,:,i), all_data_reduced(2,:,i), 'LineWidth', 2)
    hold on
end
xlabel('Time (s)')
ylabel('Dim 1 Units')
zlabel('Dim 2 Units')

xlim([-1.2, 2])


[Y,Z] = meshgrid(min(all_data_reduced(1,:,:),[],'all'):0.1:max(all_data_reduced(1,:,:),[],'all'),min(all_data_reduced(2,:,:),[],'all'):0.1:max(all_data_reduced(2,:,:),[],'all'));
X = Z*0;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'r');
s.EdgeColor = 'none';

X = Z*0+0.3;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'm');
s.EdgeColor = 'none';

X = Z*0+0.9;
s = surf(X,Y,Z,'FaceAlpha',0.2, 'FaceColor', 'b');
s.EdgeColor = 'none';

legend('3', '6', '9', 'Reward Cue', 'Delay Period', 'Reaction')
hold off

%% Plotting First Three PCs of the data in 3D coords - Just Expected Value
clc
% close all

% shuffling boolian, 1 will shuffle the Units labels
shuffle_bool = 1;

all_data_tmp = zeros(numel(Unit), nbins, 6);
all_data = zeros(2*numel(Unit), nbins, 3);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data_tmp(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
    all_data(:,:,cue_value/3) = [all_data_tmp(:,:,cue_indx-2); all_data_tmp(:,:,cue_indx-1)];
end

if shuffle_bool
    for neuron_indx = 1:numel(Unit)
        index_vec = 1:3;
        index_vec = index_vec(randperm(length(index_vec)));
        all_data(neuron_indx,:,:) = all_data(neuron_indx, :, index_vec);
    end
end

figure
all_data_reduced = zeros(3, size(all_data, 2), size(all_data, 3));
for i = 1:3
    cov_mat = cov(all_data(:,:,i));
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);

    all_data_reduced(:,:,i) = V(:, 1:3)';
    plot3(all_data_reduced(1,:,i), all_data_reduced(2,:,i), all_data_reduced(3,:,i), 'LineWidth', 2)
    hold on
end
xlabel('Dim 1 Units')
ylabel('Dim 2 Units')
zlabel('Dim 3 Units')

legend('3', '6', '9')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% CFR
% reducing units dimensinos to 1 - all conditions
clc
% close all

% shuffling boolian, 1 will shuffle the Units labels
shuffle_bool = 1;

all_data = zeros(numel(Unit), nbins, 6);

cue_indx = 1;
for cue_value = 3:3:9
    for pos = [-1, 1]
        counts_all = [];
        value = [cue_value, pos];
        for neuron_indx = 1:numel(Unit)
            data = get_condition(Unit, neuron_indx, value);
            [counts,~] = PSTH(data, window_length, nbins, centers);
            all_data(neuron_indx, :, cue_indx) = counts;
        end
        cue_indx = cue_indx+1;
    end
end
surrogate_type = 'surrogate-TNC';
model_dim = 6;
times_msk = centers>=0;

if shuffle_bool
    all_data_s = CFR(permute(all_data, [2,1,3]), surrogate_type, model_dim, times_msk);
    all_data = permute(all_data_s, [2,1,3]);
end

figure
hold on
all_data_reduced = zeros(size(all_data, 2), size(all_data, 3));

for i = 1:6
    cov_mat = cov(all_data(:,:,i)');
    [V,D] = eig(cov_mat);
    D = diag(D);
    [D, I] = sort(D, 'descend');
    D = diag(D);
    V = V(:, I);
    V(:,2:end) = 0;

    B = sqrt(inv(D));
    A = V';
    Z = B*A*all_data(:,:,i);

    all_data_reduced(:,i) = Z(1, :);
    plot(centers, all_data_reduced(:, i), 'LineWidth', 2)
end

xline(0,'r', 'Reward Cue');
xline(0.3,'m','Delay Period');
xline(0.9,'b', 'Reaction');
xlim([-1.2, 2])
ylim([min(all_data_reduced,[],'all')-1, max(all_data_reduced,[],'all')+1])
xlabel("Time (s)")
ylabel('Firing Rate (Hz)')
title("PSTH of Dimension-Reduced Activity")

box = [0 0 0.3 0.3];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'r','FaceAlpha',0.1)

box = [0.3 0.3 0.9 0.9];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'m','FaceAlpha',0.1)

box = [0.9 0.9 2 2];
boxy = [min(all_data_reduced,[],'all')-1 max(all_data_reduced,[],'all')+1 max(all_data_reduced,[],'all')+1 min(all_data_reduced,[],'all')-1];
patch(box,boxy,'b','FaceAlpha',0.1)

legend('[3 -1]', '[3 1]', '[6 -1]', '[6 1]', '[9 -1]', '[9 1]')
hold off