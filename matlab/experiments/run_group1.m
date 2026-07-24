% run_group1.m
% First validation batch: all 12 Group 1 (ADAM/sgd, T=1) runs from the
% rerun matrix. Covers both architectures and every basis path
% (monomial, Legendre d=3..6, none) -- feeds Figures 2-5 and half of
% Table 2 (the ADAM columns at T=1).
%
% Each call is wrapped in try/catch so one failure is logged and the
% batch keeps going rather than stopping cold.

runs = {
    {'ResNN', 1, 3, 'sgd', 'monomial', 'ELM'};
    {'ResNN', 1, 3, 'sgd', 'Legendre', 'ELM'};
    {'ResNN', 1, 4, 'sgd', 'Legendre', 'ELM'};
    {'ResNN', 1, 5, 'sgd', 'Legendre', 'ELM'};
    {'ResNN', 1, 6, 'sgd', 'Legendre', 'ELM'};
    {'ResNN', 1, [], 'sgd', 'none', 'ELM'};
    {'hamiltonian', 1, 3, 'sgd', 'monomial', 'ELM'};
    {'hamiltonian', 1, 3, 'sgd', 'Legendre', 'ELM'};
    {'hamiltonian', 1, 4, 'sgd', 'Legendre', 'ELM'};
    {'hamiltonian', 1, 5, 'sgd', 'Legendre', 'ELM'};
    {'hamiltonian', 1, 6, 'sgd', 'Legendre', 'ELM'};
    {'hamiltonian', 1, [], 'sgd', 'none', 'ELM'};
};

nOK = 0; nFail = 0;
for i = 1:numel(runs)
    args = runs{i};
    fprintf('\n===== Run %d/%d: %s, d=%s, %s, basis=%s =====\n', ...
        i, numel(runs), args{1}, mat2str(args{3}), args{4}, args{5});
    try
        runExperiment_v2(args{1}, args{2}, args{3}, args{4}, args{5});
        nOK = nOK + 1;
    catch ME
        nFail = nFail + 1;
        fprintf(2, '*** FAILED: %s ***\n%s\n', ME.identifier, ME.message);
    end
end

fprintf('\n===== Group 1 batch done: %d succeeded, %d failed (of %d) =====\n', nOK, nFail, numel(runs));
if nFail == 0
    fprintf('Running collectResults()...\n');
    collectResults();
end