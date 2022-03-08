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