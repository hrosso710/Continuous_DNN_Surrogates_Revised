% run_all_table2.m
% Runs every (dataset, T, optimizer) combination needed for Table 2/3 and
% Figures 2-5: 3 datasets x 3 depths x 2 optimizers = 18 groups x 12
% configs each (ResNN/hamiltonian x {monomial d3, Legendre d3-6, none}).
%
% This is the single entry point for the full MATLAB sweep -- run this
% one script to regenerate everything runConfigBatch.m can produce.
%
% Each group is independent, so if you're splitting this across separate
% SLURM jobs instead of one sitting, submit individual
%   runConfigBatch(dataset, T, opti)
% calls rather than this file.
%
% collectResults() is called once at the end (not after every group) to
% avoid rescanning the results/ directory 18 times.

groups = {
    {'ELM', 1,  'sgd'},    {'CDR', 1,  'sgd'},    {'DCR', 1,  'sgd'},    ...
    {'ELM', 1,  'GNvpro'}, {'CDR', 1,  'GNvpro'}, {'DCR', 1,  'GNvpro'}, ...
    {'ELM', 5,  'sgd'},    {'CDR', 5,  'sgd'},    {'DCR', 5,  'sgd'},    ...
    {'ELM', 5,  'GNvpro'}, {'CDR', 5,  'GNvpro'}, {'DCR', 5,  'GNvpro'}, ...
    {'ELM', 10, 'sgd'},    {'CDR', 10, 'sgd'},    {'DCR', 10, 'sgd'},    ...
    {'ELM', 10, 'GNvpro'}, {'CDR', 10, 'GNvpro'}, {'DCR', 10, 'GNvpro'}, ...
};

totalOK = 0; totalFail = 0;
for g = 1:numel(groups)
    ds = groups{g}{1}; T = groups{g}{2}; opti = groups{g}{3};
    fprintf('\n############ Batch %d/%d: %s, T=%d, %s ############\n', ...
        g, numel(groups), ds, T, opti);
    [nOK, nFail] = runConfigBatch(ds, T, opti);
    totalOK = totalOK + nOK;
    totalFail = totalFail + nFail;
end

fprintf('\n============ All Table 2 batches done: %d succeeded, %d failed (of %d) ============\n', ...
    totalOK, totalFail, totalOK + totalFail);
fprintf('Running collectResults()...\n');
collectResults();
