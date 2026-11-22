function kseries_fit(K, run, outroot)
% Fit one HMM with K states using the K7_2 options (HMM_Compute.m,
% data_modality = 'M/EEG power'), and save a small summary only: the lab
% volume has little free space, so Gamma and Xi are not written.
addpath(genpath('/imaging/astle/HMM_Dissertation/Scripts/HMM-MAR-master/'));
% serial: each parallel initialisation worker held 35-90 GB (K = 4-14)
rng(run, 'twister');

[data, T, ids] = kseries_load();

options = struct();
options.K = K;
options.verbose = 1;
options.Fs = 250;
options.order = 0;
options.zeromean = 0;
options.covtype = 'full';
options.onpower = 1;
options.useParallel = 0;   % execution only; K7_2 used 1

tic;
[hmm, Gamma, Xi, vpath, ~, ~, fehist] = hmmmar(data, T, options);
elapsed_s = toc;

% Free energy recomputed from the fitted model, as is done for K7_2, so the
% two routes can be compared on the same fit.
fe_recomputed = hmmfe(data, T, hmm, Gamma, Xi);
clear Xi

ndim = size(data{1}, 2);
means = zeros(K, ndim);
covs = zeros(K, ndim, ndim);
for k = 1:K
    means(k, :) = getMean(hmm, k);
    covs(k, :, :) = getFuncConn(hmm, k);
end
fo = getFractionalOccupancy(Gamma, T, hmm.train, 2);   % participants x K
clear Gamma
fe_final = fehist(end);
cycles = numel(fehist);
vpath = uint8(vpath);
Tsub = cellfun(@sum, T);

outdir = fullfile(outroot, sprintf('K%02d_run%d', K, run));
if ~exist(outdir, 'dir'), mkdir(outdir); end
save(fullfile(outdir, 'summary.mat'), 'K', 'run', 'means', 'covs', 'fo', 'fehist', ...
     'fe_final', 'fe_recomputed', 'cycles', 'elapsed_s', 'vpath', 'Tsub', 'ids', '-v7');
save(fullfile(outdir, 'hmm.mat'), 'hmm', '-v7');
fprintf('K=%d run=%d: free energy %.6g (recomputed %.6g), %d cycles, %.1f h\n', ...
        K, run, fe_final, fe_recomputed, cycles, elapsed_s / 3600);

end
