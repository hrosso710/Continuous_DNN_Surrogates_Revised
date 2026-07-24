function collectResults(resultsDir, tikzDir)
% collectResults  Scan every results/*.mat produced by runExperiment.m and:
%   1) build a flat summary CSV (one row per run, all metrics + provenance)
%   2) build a Table-2-shaped pivot CSV (dataset x T x basis -> ResNet/
%      Hamiltonian x ADAM/GNvpro columns of trainLoss and testLoss)
%   3) auto-export tikz data files for every run that feeds Figures 2-5
%      (ADAM/sgd, T=1, basis in {monomial,Legendre,none})
%
% Usage:
%   collectResults()                       % results/ -> tikzdata/
%   collectResults('results','tikzdata')   % explicit paths
%
% Deliberately avoids MATLAB's table/writetable/cell2table so this can be
% verified against Octave (which lacks robust table support) as well as
% real MATLAB.

if not(exist('resultsDir','var')) || isempty(resultsDir)
    resultsDir = 'results';
end
if not(exist('tikzDir','var')) || isempty(tikzDir)
    tikzDir = 'tikzdata';
end

files = dir(fullfile(resultsDir,'*.mat'));
if isempty(files)
    error('collectResults:noFiles','No .mat files found in %s', resultsDir);
end

% ---- Load every run into a flat cell array of records ----
% Columns: dataset,dynamic,basis,d,T,opti,trainLoss,valLoss,testLoss,
%          relErrTrainMean,relErrValMean,relErrTestMean,nParams,file
nRuns = numel(files);
data = cell(nRuns,14);
for i = 1:nRuns
    L = load(fullfile(files(i).folder, files(i).name));
    r = L.results;
    c = r.config;
    data(i,:) = { c.dataset, c.dynamic, c.basis, c.d, c.T, c.opti, ...
                  r.trainLoss, r.valLoss, r.testLoss, ...
                  mean(r.relErrTrain), mean(r.relErrVal), mean(r.relErrTest), ...
                  r.nParams, files(i).name };
end

colNames = {'dataset','dynamic','basis','d','T','opti', ...
            'trainLoss','valLoss','testLoss', ...
            'relErrTrainMean','relErrValMean','relErrTestMean', ...
            'nParams','file'};

summaryFile = fullfile(resultsDir,'summary_all_runs.csv');
writeCsv(summaryFile, colNames, data);
fprintf('Wrote flat summary (%d runs) -> %s\n', nRuns, summaryFile);

%% ---- Table 2 pivot: dataset x T x basis, ResNet/Hamiltonian x ADAM/GNvpro ----
% Only Legendre(d=3) and none are relevant to Table 2 (per the manuscript).
datasetCol = data(:,1); dynamicCol = data(:,2); basisCol = data(:,3);
dCol = cellfun(@(x) coerceScalarD(x), data(:,4)); 
Tcol = cell2mat(data(:,5)); optiCol = data(:,6);
metricCols = struct( ...
    'trainLoss',      cell2mat(data(:,7)), ...
    'testLoss',       cell2mat(data(:,9)), ...
    'relErrTrainMean',cell2mat(data(:,10)), ...
    'relErrValMean',  cell2mat(data(:,11)), ...
    'relErrTestMean', cell2mat(data(:,12)), ...
    'nParams',        cell2mat(data(:,13)) );

datasets = unique(datasetCol);
Ts       = unique(Tcol);

architectures = {'ResNN','hamiltonian'};
archLabels    = {'ResNet','Hamiltonian'};
optimizers    = {'sgd','GNvpro'};
optiLabels    = {'ADAM','GNvpro'};
metricNames   = {'trainLoss','testLoss','relErrTrainMean','relErrValMean','relErrTestMean','nParams'};

pivotCols = {'dataset','T','basis'};
for m = 1:numel(metricNames)
    for a = 1:numel(architectures)
        for o = 1:numel(optimizers)
            pivotCols{end+1} = sprintf('%s_%s_%s', archLabels{a}, optiLabels{o}, metricNames{m}); %#ok<AGROW>
        end
    end
end

pivotRows = {};
for di = 1:numel(datasets)
    ds = datasets{di};
    for ti = 1:numel(Ts)
        Tval = Ts(ti);
        for basisName = {'Legendre','none'}
            bn = basisName{1};
            row = {ds, Tval, bn};
            for m = 1:numel(metricNames)
                valCol = metricCols.(metricNames{m});
                for a = 1:numel(architectures)
                    for o = 1:numel(optimizers)
                        row{end+1} = lookup(ds,architectures{a},bn,Tval,optimizers{o}, ...
                            datasetCol,dynamicCol,basisCol,Tcol,optiCol,dCol,valCol); %#ok<AGROW>
                    end
                end
            end
            pivotRows(end+1,:) = row; %#ok<AGROW>
        end
    end
end

pivotFile = fullfile(resultsDir,'table2_pivot.csv');
writeCsv(pivotFile, pivotCols, pivotRows);
fprintf('Wrote Table 2 pivot -> %s\n', pivotFile);
fprintf('  (NaN entries mean that run has not been completed/saved yet)\n');

%% ---- Auto-export tikz data for Figures 2-5 (ADAM/sgd, T=1) ----
isFigRun = strcmp(optiCol,'sgd') & (Tcol==1);
figIdx = find(isFigRun);
nExported = 0;
for k = 1:numel(figIdx)
    i = figIdx(k);
    matFile = fullfile(resultsDir, data{i,14});
    tag = sprintf('%s_%s_%s_d%d', data{i,1}, data{i,2}, data{i,3}, data{i,4});
    for series = {'trainLoss','valLoss'}
        s = series{1};
        outFile = fullfile(tikzDir, sprintf('%s_%s.dat', tag, s));
        try
            exportTikzData(matFile, s, outFile);
            nExported = nExported + 1;
        catch ME
            warning('Could not export %s from %s: %s', s, matFile, ME.message);
        end
    end
end
fprintf('Exported %d tikz data files -> %s/\n', nExported, tikzDir);

end

function v = lookup(ds,dyn,basisName,Tval,opti,datasetCol,dynamicCol,basisCol,Tcol,optiCol,dCol,valCol)
mask = strcmp(datasetCol,ds) & strcmp(dynamicCol,dyn) & strcmp(basisCol,basisName) & ...
       (Tcol==Tval) & strcmp(optiCol,opti);
if strcmp(basisName,'Legendre')
    mask = mask & (dCol==3);   % Table 2 only ever shows degree-3 Legendre
end
idx = find(mask,1);
if isempty(idx)
    v = NaN;   % run not completed/saved yet -- makes gaps visible instead of erroring
else
    v = valCol(idx);
end
end

function writeCsv(filePath, colNames, rows)
fid = fopen(filePath,'w');
if fid == -1
    error('collectResults:cannotOpen','Could not open %s for writing', filePath);
end
fprintf(fid, '%s\n', strjoin(colNames,','));
for i = 1:size(rows,1)
    parts = cell(1,size(rows,2));
    for j = 1:size(rows,2)
        v = rows{i,j};
        if ischar(v)
            parts{j} = v;
        elseif isnumeric(v) && isscalar(v)
            if isnan(v)
                parts{j} = '';
            else
                parts{j} = sprintf('%.10g', v);
            end
        else
            parts{j} = mat2str(v);
        end
    end
    fprintf(fid, '%s\n', strjoin(parts,','));
end
fclose(fid);
end

function v = coerceScalarD(x)
if isempty(x)
    v = NaN;
elseif isscalar(x)
    v = x;
else
    warning('collectResults:badD','Non-scalar d encountered (%s); treating as NaN', mat2str(x));
    v = NaN;
end
end


