function surrTensor = CFR(dataTensor, surrogate_type, model_dim, times_msk)

    [targetSigmaT, targetSigmaN, targetSigmaC, M] = extractFeatures(dataTensor);
    numSurrogates = 20;
    params = [];
    params.readout_mode = 2;         % select readout mode (eg neuron mode)
    params.shfl_mode = 3;         % shuffle across tensor mode (eg condition mode)
    params.fix_mode = 2;         % shuffle per mode (shuffle for each neuron independently)

    if strcmp(surrogate_type, 'surrogate-T')
        params.margCov{1} = targetSigmaT;
        params.margCov{2} = [];
        params.margCov{3} = [];
        params.meanTensor = M.T;
    elseif strcmp(surrogate_type, 'surrogate-TN')
        params.margCov{1} = targetSigmaT;
        params.margCov{2} = targetSigmaN;
        params.margCov{3} = [];
        params.meanTensor = M.TN;
    elseif strcmp(surrogate_type, 'surrogate-TNC')
        params.margCov{1} = targetSigmaT;
        params.margCov{2} = targetSigmaN;
        params.margCov{3} = targetSigmaC;
        params.meanTensor = M.TNC; 
    else
        error('please specify a correct surrogate type') 
    end


    R2_surr = nan(numSurrogates, 1);
    for i = 1:numSurrogates
        fprintf('surrogate %d from %d, ', i, numSurrogates)
        [surrTensor] = sampleCFR(dataTensor, params);       % generate CFR random surrogate data.
        [R2_surr(i)] = summarizeLDS(surrTensor(times_msk, :, :), model_dim, false);
    end

end