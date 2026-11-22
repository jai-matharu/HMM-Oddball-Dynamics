function [data, T, ids] = kseries_load()
% Load the 39 participants exactly as HMM_Compute.m did for K7_2: every
% data_*.mat in Preproc_Outputs in dir() order, minus 99114 (position 32).
datdir = '/imaging/astle/HMM_Dissertation/Outputs/Preproc_Outputs/';
datfiles = dir([datdir 'data_*.mat']);
data = {}; T = {}; ids = {};
for f = 1:numel(datfiles)
    parts = split(datfiles(f).name, '_');
    id = erase(parts{2}, '.mat');
    if strcmp(id, '99114'), continue; end
    d = load([datdir datfiles(f).name]);
    data{end+1} = d.data.X; %#ok<AGROW>
    t = load([datdir 'T_' id '.mat']);
    T{end+1} = t.T; %#ok<AGROW>
    ids{end+1} = id; %#ok<AGROW>
end
assert(numel(data) == 39, 'expected 39 participants, got %d', numel(data));
fprintf('loaded %d participants, %d samples\n', numel(data), sum(cellfun(@sum, T)));
end
