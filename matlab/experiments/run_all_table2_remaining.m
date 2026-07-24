% run_all_table2_remaining.m
% Runs every remaining (dataset, T, opti) combo needed to complete Table 2:
% Group 1 (T=1, ADAM) is already done for ELM/CDR/DCR. This drives Groups
% 2-6 (T=1 GNvpro; T=5 and T=10, both ADAM and GNvpro) across all three
% datasets -- 15 batches x 12 configs = 180 runs total.
%
% Intended for running everything in one sitting on a single allocation.
% If you're splitting this across separate SLURM jobs instead, submit the
% individual run_group{N}_{dataset}.m scripts rather than this file --
% each is independent and only needs runConfigBatch.m + runExperiment_v2.m
% on the path.
%
% collectResults() is called once at the end here (not after every group)
% to avoid rescanning the results/ directory 15 times; the per-group
% scripts call it themselves when run standalone.

groups = {
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

fprintf('\n============ All remaining Table 2 batches done: %d succeeded, %d failed (of %d) ============\n', ...
    totalOK, totalFail, totalOK + totalFail);
fprintf('Running collectResults()...\n');
collectResults();