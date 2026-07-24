function [nOK, nFail] = runConfigBatch(dataset, T, opti)
% runConfigBatch  Run the standard 12-config sweep (ResNN/hamiltonian x
%                 {monomial d3, Legendre d3-6, none}) for one dataset, one
%                 depth T, and one optimizer.
%
% Factored out of run_group1_CDR.m's body so that T/opti/dataset can vary
% without duplicating the 12-row config list in every launcher script --
% there is now exactly one place (this file) where that list lives.
%
% Usage:
%   runConfigBatch('ELM', 1, 'GNvpro')
%   runConfigBatch('DCR', 5, 'sgd')
%
% Does NOT call collectResults() itself -- callers decide when to
% aggregate (e.g. after all groups for a SLURM array have finished, or
% after every single group, matching run_group1_CDR.m's convention).
%
% Watch the console for "*** FAILED ***" markers -- with try/catch around
% each call, a failure gets logged and the batch keeps going rather than
% stopping, which is easy to miss scrolling past 10000 epochs of ADAM
% output (or many GNvpro work units) per run.

if not(exist('dataset','var')) || isempty(dataset)
    error('runConfigBatch:missingDataset','dataset is required (ELM, CDR, or DCR)');
end
if not(exist('T','var')) || isempty(T)
    error('runConfigBatch:missingT','T is required');
end
if not(exist('opti','var')) || isempty(opti)
    error('runConfigBatch:missingOpti','opti is required (sgd or GNvpro)');
end

% {dynamic, d, basis} -- same 12 configs as run_group1_CDR.m, generalized
% over T/opti/dataset via the function arguments instead of being baked
% into the tuple.
archAndBasis = {
    {'ResNN',       3,  'monomial'};
    {'ResNN',       3,  'Legendre'};
    {'ResNN',       4,  'Legendre'};
    {'ResNN',       5,  'Legendre'};
    {'ResNN',       6,  'Legendre'};
    {'ResNN',       [], 'none'};
    {'hamiltonian', 3,  'monomial'};
    {'hamiltonian', 3,  'Legendre'};
    {'hamiltonian', 4,  'Legendre'};
    {'hamiltonian', 5,  'Legendre'};
    {'hamiltonian', 6,  'Legendre'};
    {'hamiltonian', [], 'none'};
};

nOK = 0; nFail = 0;
for i = 1:numel(archAndBasis)
    cfg = archAndBasis{i};
    dynamic = cfg{1}; d = cfg{2}; basis = cfg{3};
    fprintf('\n===== [%s, T=%g, %s] Run %d/%d: %s, d=%s, basis=%s =====\n', ...
        dataset, T, opti, i, numel(archAndBasis), dynamic, mat2str(d), basis);
    try
        runExperiment_v2(dynamic, T, d, opti, basis, dataset);
        nOK = nOK + 1;
    catch ME
        nFail = nFail + 1;
        fprintf(2, '*** FAILED: %s ***\n%s\n', ME.identifier, ME.message);
    end
end

fprintf('\n===== [%s, T=%g, %s] batch done: %d succeeded, %d failed (of %d) =====\n', ...
    dataset, T, opti, nOK, nFail, numel(archAndBasis));

end