% run_group1_CDR.m
% CDR version of Group 1 (ADAM/sgd, T=1) -- same 12 configs as ELM's
% run_group1.m, with 'CDR' appended as the dataset argument.
%
% ResNN/monomial/d3 is commented out below since your standalone run
% already produced results/CDR_ResNN_monomial_d3_T1_sgd.mat -- uncomment
% it if you want to redo it anyway (e.g. to confirm reproducibility).
%
% Watch the console for "*** FAILED ***" markers -- with try/catch
% around each call, a failure gets logged and the batch keeps going
% rather than stopping, which is easy to miss scrolling past 10000
% epochs of ADAM output per run.

runs = {
    % {'ResNN', 1, 3, 'sgd', 'monomial', 'CDR'};   % already completed standalone -- uncomment to redo
    {'ResNN', 1, 3, 'sgd', 'Legendre', 'CDR'};
    {'ResNN', 1, 4, 'sgd', 'Legendre', 'CDR'};
    {'ResNN', 1, 5, 'sgd', 'Legendre', 'CDR'};
    {'ResNN', 1, 6, 'sgd', 'Legendre', 'CDR'};
    {'ResNN', 1, [], 'sgd', 'none', 'CDR'};
    {'hamiltonian', 1, 3, 'sgd', 'monomial', 'CDR'};
    {'hamiltonian', 1, 3, 'sgd', 'Legendre', 'CDR'};
    {'hamiltonian', 1, 4, 'sgd', 'Legendre', 'CDR'};
    {'hamiltonian', 1, 5, 'sgd', 'Legendre', 'CDR'};
    {'hamiltonian', 1, 6, 'sgd', 'Legendre', 'CDR'};
    {'hamiltonian', 1, [], 'sgd', 'none', 'CDR'};
};

nOK = 0; nFail = 0;
for i = 1:numel(runs)
    args = runs{i};
    fprintf('\n===== Run %d/%d: %s, d=%s, %s, basis=%s, dataset=%s =====\n', ...
        i, numel(runs), args{1}, mat2str(args{3}), args{4}, args{5}, args{6});
    try
        runExperiment_v2(args{1}, args{2}, args{3}, args{4}, args{5}, args{6});
        nOK = nOK + 1;
    catch ME
        nFail = nFail + 1;
        fprintf(2, '*** FAILED: %s ***\n%s\n', ME.identifier, ME.message);
    end
end

fprintf('\n===== Group 1 (CDR) batch done: %d succeeded, %d failed (of %d) =====\n', nOK, nFail, numel(runs));
if nFail == 0
    fprintf('Running collectResults()...\n');
    collectResults();
end