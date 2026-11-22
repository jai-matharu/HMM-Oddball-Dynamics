function kseries_k7_2_fe(outroot)
% Free energy of the analysed K7_2 fit, recomputed from its saved Gamma and Xi:
% its fehist_k7.mat holds GammaInit (the fifth output of hmmmar), not the
% free-energy history, so the value was never saved. Also extracts the state
% means with getMean, to check against state_maps_K7_2.npz.
addpath(genpath('/imaging/astle/HMM_Dissertation/Scripts/HMM-MAR-master/'));
kseries_pool();
src = '/imaging/astle/HMM_Dissertation/Outputs/HMM_Outputs/K7_2/';
[data, T, ids] = kseries_load();
load([src 'hmm_k7.mat'], 'hmm');
load([src 'Gamma_k7.mat'], 'Gamma');
load([src 'Xi_k7.mat'], 'Xi');
fe_recomputed = hmmfe(data, T, hmm, Gamma, Xi);
clear Xi
K = hmm.K; ndim = size(data{1}, 2);
means = zeros(K, ndim); covs = zeros(K, ndim, ndim);
for k = 1:K
    means(k, :) = getMean(hmm, k);
    covs(k, :, :) = getFuncConn(hmm, k);
end
fo = getFractionalOccupancy(Gamma, T, hmm.train, 2);
clear Gamma
outdir = fullfile(outroot, 'K07_K7_2');
if ~exist(outdir, 'dir'), mkdir(outdir); end
save(fullfile(outdir, 'summary.mat'), 'K', 'means', 'covs', 'fo', 'fe_recomputed', 'ids', '-v7');
fprintf('K7_2: recomputed free energy %.6g\n', fe_recomputed);
delete(gcp('nocreate'));
end
