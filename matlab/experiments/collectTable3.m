function collectTable3(resultsDir)
% collectTable3  Build Table 3 (DCR, Hamiltonian, T=1, ADAM/sgd):
%   training relative error and dynamics-only parameter count vs.
%   Legendre degree (3-6), plus the non-parameterized baseline row.
%
% NOTE ON SCOPE: unlike collectResults.m's Table 2 pivot (which reads
% relErrTestMean and the combined nParams = numel(thOpt)+numel(WOpt)
% field), Table 3 uses:
%   - TRAINING relative error (relErrTrain), not test -- confirmed 2026-07
%   - dynamics-only parameter count, i.e. numel(thOpt) alone, recomputed
%     directly from thOpt rather than trusting the saved `nParams` field,
%     since that field bundles in numel(WOpt) and Table 3's caption
%     explicitly excludes the readout matrix W.
%   - DeltaError = relErrTrainMean(d) - relErrTrainMean(baseline), i.e.
%     ABSOLUTE difference -- confirmed 2026-07. Negative = parameterized
%     model beats the baseline.
%
% Usage:
%   collectTable3()            % results/ -> results/table3_pivot.csv
%   collectTable3('results')   % explicit path
 
if not(exist('resultsDir','var')) || isempty(resultsDir)
    resultsDir = 'results';
end
 
dataset = 'DCR';
dynamic = 'hamiltonian';
T       = 1;
opti    = 'sgd';   % ADAM
 
degrees = [3 4 5 6];
 
% ---- Load baseline (basis='none', d=0) ----
baselineTag  = sprintf('%s_%s_none_d0_T%g_%s', dataset, dynamic, T, opti);
baselineFile = fullfile(resultsDir, [baselineTag '.mat']);
if not(exist(baselineFile,'file'))
    error('collectTable3:noBaseline', ...
        'Baseline file not found: %s -- run runExperiment_v2(''%s'',%g,[],''%s'',''none'',''%s'') first', ...
        baselineFile, dynamic, T, opti, dataset);
end
L = load(baselineFile);
baselineErr    = mean(L.results.relErrTrain);
baselineParams = numel(L.results.thOpt);
fprintf('Baseline [%s]: relErrTrainMean=%.6f, nParams(theta only)=%d\n', ...
    baselineTag, baselineErr, baselineParams);
 
% ---- Load each degree ----
nRows = numel(degrees) + 1;  % + baseline row
data = cell(nRows, 5);  % degree, basis, relErrTrainMean, nParamsTheta, DeltaError
 
% baseline row first
data(1,:) = {0, 'none', baselineErr, baselineParams, 0};
 
for i = 1:numel(degrees)
    d = degrees(i);
    tag  = sprintf('%s_%s_Legendre_d%d_T%g_%s', dataset, dynamic, d, T, opti);
    file = fullfile(resultsDir, [tag '.mat']);
    if not(exist(file,'file'))
        warning('collectTable3:missingRun','Missing run for degree %d: %s -- leaving NaN', d, file);
        data(i+1,:) = {d, 'Legendre', NaN, NaN, NaN};
        continue
    end
    L = load(file);
    relErrTrainMean = mean(L.results.relErrTrain);
    nParamsTheta    = numel(L.results.thOpt);
 
    deltaError = relErrTrainMean - baselineErr;   % absolute; negative = parameterized wins
 
    data(i+1,:) = {d, 'Legendre', relErrTrainMean, nParamsTheta, deltaError};
 
    fprintf('d=%d [%s]: relErrTrainMean=%.6f, nParams(theta only)=%d, DeltaError=%+.6f\n', ...
        d, tag, relErrTrainMean, nParamsTheta, deltaError);
end
 
% ---- Write CSV ----
colNames = {'degree','basis','relErrTrainMean','nParamsTheta','DeltaError'};
outFile = fullfile(resultsDir, 'table3_pivot.csv');
fid = fopen(outFile,'w');
if fid == -1
    error('collectTable3:cannotOpen','Could not open %s for writing', outFile);
end
fprintf(fid, '%s\n', strjoin(colNames,','));
for i = 1:size(data,1)
    row = data(i,:);
    parts = cell(1,numel(row));
    for j = 1:numel(row)
        v = row{j};
        if ischar(v)
            parts{j} = v;
        elseif isnan(v)
            parts{j} = '';
        else
            parts{j} = sprintf('%.10g', v);
        end
    end
    fprintf(fid, '%s\n', strjoin(parts,','));
end
fclose(fid);
fprintf('\nWrote Table 3 pivot -> %s\n', outFile);
 
end