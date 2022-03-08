
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