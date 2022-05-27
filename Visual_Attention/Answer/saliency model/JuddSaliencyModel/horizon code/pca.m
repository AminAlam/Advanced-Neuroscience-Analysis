function [pc, latent] = pca(features, N)

%fm = features - repmat(mean(features, 2), 1, size(features,2));
%X= double(fm*fm'); 
%[pc, latent] = eigs(X, N);

[pc, latent] = eigs(cov(X), N);

