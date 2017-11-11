 clear;

rng('default');
vl_setupnn;

%%% INITIALIZE CNN
netparams.nsigs     = 2;
netparams.N         = 256;
netparams.wconvsize = 8;
netparams.marray    = -96:95;
%netparams.marray    = -floor(2*(netparams.N-netparams.wconvsize+1)/3):floor(2*(netparams.N-netparams.wconvsize+1)/3 -1 );
netparams.sigma     = 0.01;
netparams.nwin      = 64;
netparams.batch_size = 128;
netparams.f         =  0.01;

refnet = tRefNet_init(netparams);
%my_simplenn_display(refnet);

medMatfilename = sprintf('/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/AUTOTEST_N%d_NW%d_XPAN10_medleyVBRdataset.mat',netparams.N,netparams.nwin);
%medMatfilename = sprintf('/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/REFTEST_N%d_NW%d_XPAN10_medleyVBRdataset.mat',netparams.N,netparams.nwin);


N = netparams.N;
nwin = netparams.nwin;
nsigs = netparams.nsigs;
batch_size = netparams.batch_size;

% Gather comb class
fid = fopen(medMatfilename,'r');
ncombs = fread(fid,1,'int32');
combClass = int32(fread(fid,ncombs,'int32'));
fclose(fid);

offset = ncombs*4;

m = memmapfile(medMatfilename,        ...
'Offset', offset,                ...
'Format', {                    ...
'single',  [N nwin nsigs], 'vbmat'; ...
'int32', [1 1], 'ref'},  ...
'Writable', true);

id = find(combClass);

trainOpts.train = id(randi(numel(id)-1,4096,1));
trainOpts.val = id(randi(numel(id)-1,1024,1));

prefix = 'RANDINIT_AUTOTEST2';
%prefix = 'AUTOTEST4_WL1';

% Train
%trainOpts.gpus = [] ;
trainOpts.gpus = [1] ;
trainOpts.batchSize = netparams.batch_size ;
trainOpts.plotDiagnostics = true ;
trainOpts.plotStatistics = true;
trainOpts.numEpochs = 1000 ;
trainOpts.epochSize = inf ;
trainOpts.numSubBatches = 1 ;
trainOpts.learningRate = [0.01 * ones(1,25), 0.007 * ones(1,100), 0.004 * ones(1,200), 0.002 * ones(1,500)] ;
trainOpts.momentum = 0.8 ;
trainOpts.weightDecay = 0.001 ;
trainOpts.errorFunction = 'multiclass' ;
trainOpts.prefetch = false;
trainOpts.continue = false;
trainOpts.profile = false;
trainOpts.expDir = sprintf('/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/mat_conv_data/%s_NT%d_NV%d_bsize%d_N%d_NW%d', prefix, numel(trainOpts.train), numel(trainOpts.val), netparams.batch_size, netparams.N, netparams.nwin)


fprintf ('Total available inputs %d. Train %d | Eval %d\n', numel(id), numel(trainOpts.train), numel(trainOpts.val));

assert(min(size(m.Data(1).vbmat) == [netparams.N netparams.nwin netparams.nsigs]));

[refnet, stats] = tRefNet_train(refnet, m, @getBatch, trainOpts) ;

% inputs.data = gpuArray(inputs.data);
% res = my_simplenn(refnet,inputs.data);
% 
% matout = res(end).x;
% pout =  (matout - bmref) ./ (matout .* (1 - matout));
% 
% res = my_simplenn(refnet,inputs.data,pout);

