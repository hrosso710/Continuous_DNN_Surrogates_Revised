% run_group1_DCR.m
% DCR version of Group 1 (ADAM/sgd, T=1) -- same 12 configs as ELM's
% run_group1.m and CDR's run_group1_CDR.m, with 'DCR' appended as the
% dataset argument.
%
% Unlike run_group1_CDR.m, no DCR run has been completed standalone yet,
% so all 12 configs run here (nothing commented out). If you've already
% run ResNN/monomial/d3 standalone as results/DCR_ResNN_monomial_d3_T1_sgd.mat,
% comment that line out the same way run_group1_CDR.m does.
%
% Watch the console for "*** FAILED ***" markers -- with try/catch
% around each call, a failure gets logged and the batch keeps going
% rather than stopping, which is easy to miss scrolling past 10000
% epochs of ADAM output per run.

runs = {
    {'ResNN', 1, 3, 'sgd', 'monomial', 'DCR'};
    {'ResNN', 1, 3, 'sgd', 'Legendre', 'DCR'};
    %{'ResNN', 1, 4, 'sgd', 'Legendre', 'DCR'};
    %{'ResNN', 1, 5, 'sgd', 'Legendre', 'DCR'};
    %{'ResNN', 1, 6, 'sgd', 'Legendre', 'DCR'};
    {'ResNN', 1, [], 'sgd', 'none', 'DCR'};
    {'hamiltonian', 1, 3, 'sgd', 'monomial', 'DCR'};
    {'hamiltonian', 1, 3, 'sgd', 'Legendre', 'DCR'};
    %{'hamiltonian', 1, 4, 'sgd', 'Legendre', 'DCR'};
    %{'hamiltonian', 1, 5, 'sgd', 'Legendre', 'DCR'};
    %{'hamiltonian', 1, 6, 'sgd', 'Legendre', 'DCR'};
    {'hamiltonian', 1, [], 'sgd', 'none', 'DCR'};
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

fprintf('\n===== Group 1 (DCR) batch done: %d succeeded, %d failed (of %d) =====\n', nOK, nFail, numel(runs));
if nFail == 0
    fprintf('Running collectResults()...\n');
    collectResults();
end