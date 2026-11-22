function kseries_pool()
% Local parallel pool on the allocated node, with job storage private to this
% Slurm job so that concurrent pools do not collide.
n = str2double(getenv('SLURM_CPUS_PER_TASK'));
if isnan(n), n = 1; end
c = parcluster('local');
store = fullfile('/imaging/astle/natalia/oddball_jobs/kseries/pool', getenv('SLURM_JOB_ID'));
if ~exist(store, 'dir'), mkdir(store); end
c.JobStorageLocation = store;
c.NumWorkers = n;
if isempty(gcp('nocreate')), parpool(c, n); end
end
