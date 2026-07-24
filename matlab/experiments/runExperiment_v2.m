function results = runExperiment_v2(dynamic,T,d,opti,basis,dataset)
% runExperiment  Train a DTO (discretize-then-optimize) ResNet/Hamiltonian
%                network on a surrogate task and save everything needed
%                to populate Table 2/3 and Figures 2-5.
%
% dynamic : one of 'antiSym-ResNN','ResNN','hamiltonian','leapfrog'
% T       : final time
% d       : degree of the polynomial basis (ignored for basis='none')
% opti    : one of 'sgd' (ADAM), 'GNvpro'
% basis   : one of 'monomial','Legendre','qr','none' ; default = 'monomial'
% dataset : one of 'ELM','CDR','DCR' ; default = 'ELM'
%
% Returns `results`, a struct with everything that gets saved to disk, so
% this can also be called interactively without relying on disk I/O.

resultsDir = 'results';
if not(exist(resultsDir,'dir')); mkdir(resultsDir); end

if not(exist('basis','var')) || isempty(basis)
    basis = 'monomial';
end
if not(exist('dataset','var')) || isempty(dataset)
    dataset = 'ELM';
end

rng(42);
switch dataset
    case 'ELM'
        [Yt,Ct,Yv,Cv,Ytest,Ctest] = setupNNERDS();
    case 'CDR'
        [Yt,Ct,Yv,Cv,Ytest,Ctest] = setupCDR();   % ignores optional nTrain/nVal/seed args; defaults match the paper's 400/200 split
    case 'DCR'
        [Yt,Ct,Yv,Cv,Ytest,Ctest] = setupDCR();   % ignores optional nTrain/nVal/seed args; defaults match the paper's 400/200 split (same convention as CDR above)
    otherwise
        error('runExperiment:unknownDataset','Unknown dataset: %s',dataset);
end

rng(20);

nt      = 12;   % number of time steps (will not prolongate with nt=1)
nc      = 15;   % number of channels (width) -- tuned for ELM; revisit per-dataset once CDR/DCR results come in

% regularization
alpha1 = 1e-5; % theta
alpha2 = 1e-3; % W

% setup networkrel

% first block
block1 = NN({singleLayer(dense([nc,size(Yt,1)],'Bin',eye(nc)))});

% second block (ResNN, keeps size fixed)
switch dynamic
    case 'ResNN'
        K       = dense([nc,nc]);
        layer   = singleLayer(K,'Bin',eye(nc));

        if strcmpi(basis,'none')
            block2 = ResNN(layer,nt,T/nt);   % no 'A' override -> defaults to eye(nt): non-parameterized baseline
        else
            A = getPolynomialBasis(d,T,nt,basis);   % nt x (d+1); dispatches correctly on 'basis'
            block2  = ResNN(layer,nt,T/nt,'A',A');
        end
    case 'antiSym-ResNN'
        K       = getDenseAntiSym([nc,nc]);
        layer   = singleLayer(K,'Bin',eye(nc));
        tY      = linspace(0,T,nt);
        block2  = ResNNrk4(layer,tY,tY);
    case 'leapfrog'
        K      = dense([nc,nc]);
        layer  = doubleSymLayer(K,'Bout',eye(nc));
        if strcmpi(basis,'none')
            block2 = LeapFrogNN(layer,nt,T/nt);   % non-parameterized baseline
        else
            A = getPolynomialBasis(d,T,nt,basis);   % nt x (d+1)
            block2 = LeapFrogNN(layer,nt,T/nt,'A',A');
        end
    case 'hamiltonian'
        K       = dense([nc,nc]);
        if strcmpi(basis,'none')
            block2 = HamiltonianNN(@tanhActivation,K,eye(nc),nt,T/nt);   % non-parameterized baseline
        else
            A = getPolynomialBasis(d,T,nt,basis);   % nt x (d+1)
            block2  = HamiltonianNN(@tanhActivation,K,eye(nc),nt,T/nt,'A',A');
        end
    otherwise
        error('Example %s not yet implemented',dynamic);
end

% combine both blocks
net = Meganet({block1,block2});

if strcmpi(basis,'none')
    d = 0;   % normalize for clean filenames/provenance; degree is not meaningful for the non-parameterized baseline
end

% setup regression loss
pLoss = regressionLoss();

startTime = tic;
switch opti
    case 'sgd'
        opt = sgd('nesterov',false,'ADAM',true,'miniBatch',32,'out',1,'lossTol',-Inf);   % was 0.01 -- all runs were hitting this absolute-loss threshold at nearly the same value (~0.0097-0.0100), which erased differentiation between configs even though relative error was still ~700%; disabled so maxEpochs is the real stopping criterion
        opt.learningRate = 0.001;
        opt.maxEpochs    = 10000;

        th0 = initTheta(net);

        % ---- FIX (diagnosed from comparing WOpt norms between sgd and GNvpro
        % runs on identical configs): the old code initialized W ~ randn(...)
        % (unit variance) and optimized it jointly with theta under ADAM with
        % NO regularization on either variable. W never meaningfully moved
        % from its random initialization scale in 10,000 epochs, so
        % predictions ended up scaled ~1000x too large relative to the true
        % targets -- absolute MSE stayed deceptively small (dominated by
        % whichever samples happened to land close by chance) while
        % per-sample relative L2 error blew up to 400,000%+ on DCR. GNvpro
        % never had this problem because its variable-projection step solves
        % for W in closed form (regularized by alpha2) every iteration.
        %
        % Fix has two parts:
        %  (1) Warm-start W via the same closed-form ridge solve GNvpro's
        %      VarPro step effectively performs, using the network's output
        %      at the *initial* theta, instead of an arbitrary random draw.
        %  (2) Regularize the ADAM objective with the same alpha1/alpha2
        %      weights GNvpro's dnnVarProRegressionObjFctn uses, so ADAM
        %      optimizes the same regularized problem GNvpro does instead of
        %      an unregularized one that has no pressure to keep W small.
        %
        % NOTE ON PART (2): tikhonovReg/opEye below follow the standard
        % Meganet.m regularizer convention (pRegTheta/pRegW arguments to
        % dnnObjFctn), but this codebase has several custom extensions
        % (HamiltonianNN, LeapFrogNN, dnnVarProRegressionObjFctn) so I have
        % NOT verified this exact constructor signature against your
        % installed toolbox. If tikhonovReg/opEye aren't the right names (or
        % take arguments in a different order) MATLAB will fail loudly with
        % an "Undefined function" error right here -- if that happens, grep
        % your Meganet install for how pRegTheta/pRegW get built elsewhere
        % (dnnVarProRegressionObjFctn.m is a good place to look, since it
        % already builds an alpha1/alpha2-weighted regularizer internally)
        % and send me the actual signature so I can correct this line.
        pRegTheta = tikhonovReg(opEye(numel(th0)), alpha1);

        YNt0 = forwardProp(net,th0,Yt);                              % network output at initial theta
        Xt0  = [YNt0; ones(1,size(YNt0,2))];                         % same bias-row convention used below for relErr
        W0   = (Ct*Xt0') / (Xt0*Xt0' + alpha2*eye(size(Xt0,1)));     % ridge-regularized least squares, matching GNvpro's alpha2
        W    = W0;

        pRegW = tikhonovReg(opEye(numel(W)), alpha2);

        fctn = dnnObjFctn(net,pRegTheta,pLoss,pRegW,Yt,Ct);
        fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);   % val objective stays unregularized -- used only for early-stopping/monitoring, not optimized directly

        [thW,his]    = solve(opt,fctn,[th0;W(:)],fval);
        [thOpt,WOpt] = split(fctn,thW);   % branch-local: sgd's own W, not overwritten later

    case 'GNvpro'
        opt              = trnewton();
        opt.linSol       = GMRES('m',20,'tol',1e-2);
        opt.out          = 1;
        opt.maxIter      = Inf;
        opt.maxWorkUnits = 4000;
        opt.atol         = 1e-16;
        opt.rtol         = 1e-16;

        fctn = dnnVarProRegressionObjFctn(net,pLoss,Yt,Ct,'alpha1',alpha1,'alpha2',alpha2);
        fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

        th0 = initTheta(net);
        [thOpt,his]   = solve(opt,fctn,th0,fval);
        [~,para]      = eval(fctn,thOpt);       % VarPro-specific: para.W only valid here
        WOpt          = reshape(para.W,size(Ct,1),[]);

    otherwise
        error('optimizer %s not recognized',opti);
end
endTime = toc(startTime);
fprintf('Elapsed Time is %s seconds.\n',num2str(endTime));

%% ---- Optimizer-agnostic loss/error evaluation (used for Table 2) ----
% Computed the same way regardless of whether training used sgd/ADAM or
% GNvpro, so train/val/test numbers are directly comparable across the
% optimizer column in Table 2 (does not rely on his.his column indices,
% which are not guaranteed consistent between the two solvers).

evalFctn = @(Y,C) dnnObjFctn(net,[],pLoss,[],Y,C);
[trainLoss,~] = eval(evalFctn(Yt,Ct), [thOpt(:);WOpt(:)]);
[valLoss,~]   = eval(evalFctn(Yv,Cv), [thOpt(:);WOpt(:)]);
[testLoss,~]  = eval(evalFctn(Ytest,Ctest), [thOpt(:);WOpt(:)]);

%% ---- Relative errors (existing metric, unchanged) ----
YNt = forwardProp(net,thOpt,Yt);
WYt = reshape(WOpt,size(Ct,1),[]) * [YNt; ones(1,size(YNt,2))];
relErrTrain = sqrt(sum((WYt - Ct).^2,1)) ./ sqrt(sum(Ct.^2,1));

YNv = forwardProp(net,thOpt,Yv);
WYv = reshape(WOpt,size(Ct,1),[]) * [YNv; ones(1,size(YNv,2))];
relErrVal = sqrt(sum((WYv - Cv).^2,1)) ./ sqrt(sum(Cv.^2,1));

YNtest = forwardProp(net,thOpt,Ytest);
WYtest = reshape(WOpt,size(Ct,1),[]) * [YNtest; ones(1,size(YNtest,2))];
relErrTest = sqrt(sum((WYtest - Ctest).^2,1)) ./ sqrt(sum(Ctest.^2,1));

fprintf('%-8smean\t+/-std\t\tmin\tmax\n','')
fprintf('Train:\t%0.4f\t+/-%0.4f\t%0.4f\t%0.4f\n',mean(relErrTrain),std(relErrTrain),min(relErrTrain),max(relErrTrain));
fprintf('Val:\t%0.4f\t+/-%0.4f\t%0.4f\t%0.4f\n',mean(relErrVal),std(relErrVal),min(relErrVal),max(relErrVal));
fprintf('Test:\t%0.4f\t+/-%0.4f\t%0.4f\t%0.4f\n',mean(relErrTest),std(relErrTest),min(relErrTest),max(relErrTest));

%% ---- Parameter count (for Table 3) ----
nParams = numel(thOpt) + numel(WOpt);

%% ---- Assemble everything into one struct ----
config = struct('dataset',dataset,'dynamic',dynamic,'T',T,'d',d,'opti',opti, ...
                 'basis',basis,'nt',nt,'nc',nc,'alpha1',alpha1,'alpha2',alpha2, ...
                 'elapsedSeconds',endTime);

results = struct();
results.config      = config;
results.net         = net;
results.thOpt       = thOpt;
results.WOpt        = WOpt;
results.his         = his;              % raw solver history, kept for provenance
results.trainLoss   = trainLoss;
results.valLoss     = valLoss;
results.testLoss    = testLoss;
results.relErrTrain = relErrTrain;
results.relErrVal   = relErrVal;
results.relErrTest  = relErrTest;
results.nParams     = nParams;

%% ---- Save to disk: one .mat with everything, plus a plotting-ready CSV ----
tag = sprintf('%s_%s_%s_d%d_T%g_%s', config.dataset, dynamic, basis, d, T, opti);
matFile = fullfile(resultsDir, [tag '.mat']);
save(matFile,'results');
fprintf('Saved results to %s\n', matFile);

% Plain-text dump of the per-iteration history for tikz/pgfplots.
% his.his columns are solver-specific; we dump the whole matrix with a
% generic header rather than guessing which column is "training loss" vs
% "validation loss" -- verify column meaning against hisNames(fctn) before
% wiring this into a tikz \addplot, since sgd and GNvpro histories are not
% guaranteed to share the same column layout.
try
    hisFile = fullfile(resultsDir, [tag '_history.txt']);
    hisMat  = his.his;
    fid = fopen(hisFile,'w');
    fprintf(fid, '%% raw his.his matrix for %s -- verify column meaning via hisNames(fctn) before use\n', tag);
    fmt = [repmat('%g\t',1,size(hisMat,2)-1) '%g\n'];
    for row = 1:size(hisMat,1)
        fprintf(fid, fmt, hisMat(row,:));
    end
    fclose(fid);
    fprintf('Saved raw history to %s\n', hisFile);
catch ME
    warning('Could not export history matrix: %s', ME.message);
end

end