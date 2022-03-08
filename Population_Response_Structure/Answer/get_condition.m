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