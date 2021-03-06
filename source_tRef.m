 clear;

rng('default');
vl_setupnn;

%%% INITIALIZE CNN
netparams.nsigs     = 2;
netparams.N         = 256;
netparams.wconvsize = 8;
netparams.marray    = -80:79;
%netparams.marray    = -floor(2*(netparams.N-netparams.wconvsize+1)/3):floor(2*(netparams.N-netparams.wconvsize+1)/3 -1 );
netparams.sigma     = 0.1;
netparams.nwin      = 64;
netparams.f         =  0.02;
netparams.batch_size = 64;

refnet = tRefNet_init(netparams);
%my_simplenn_display(refnet);

medMatfilename = sprintf('/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/REFTEST_N%d_NW%d_XPAN10_medleyVBRdataset.mat',netparams.N,netparams.nwin);
%medMatfilename = sprintf('/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/REFTEST_N%d_NW%d_XPAN10_medleyVBRdataset.mat',netparams.N,netparams.nwin);


N = netparams.N;
nwin = netparams.nwin;
nsigs = netparams.nsigs;
batch_size = netparams.batch_size;

% Gather instrument names from matfile
[p,n,e] = fileparts(medMatfilename);
cinfofile = [p,n,'__cinfo',e];

load(cinfofile);

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

db.m = m;
db.allcomb = allcomb;

id = int32(find(combClass == 4));
trainOpts.train = id(randi(numel(id)-1,512,1));
id2 = setdiff(id,trainOpts.train);
trainOpts.val = id2(randi(numel(id2)-1,256,1));

prefix = 'REFTEST_BNORM';
%prefix = 'AUTOTEST4_WL1';

% Train
%trainOpts.gpus = [] ;
trainOpts.gpus = [1] ;
trainOpts.batchSize = netparams.batch_size ;
trainOpts.plotDiagnostics = false ;
trainOpts.plotStatistics = true ;
trainOpts.plotInst = true ;
trainOpts.numEpochs = 200 ;
trainOpts.epochSize = inf ;
trainOpts.numSubBatches = 1 ;
trainOpts.learningRate = 1./( 50 + exp( 0.05 * (1:trainOpts.numEpochs) ) ); % [0.01 * ones(1,25), 0.007 * ones(1,100), 0.004 * ones(1,200), 0.002 * ones(1,500)] ;
trainOpts.momentum = 0.8 ;
trainOpts.weightDecay = 0.00 ;
trainOpts.errorFunction = 'instError' ;
trainOpts.prefetch = false;
trainOpts.continue = true;
trainOpts.profile = false;                                                                                                                                                                                       
trainOpts.expDir = sprintf('/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/mat_conv_data/%s_NT%d_NV%d_bsize%d_N%d_NW%d', prefix, numel(trainOpts.train), numel(trainOpts.val), netparams.batch_size, netparams.N, netparams.nwin)


fprintf ('Total available inputs %d. Train %d | Eval %d\n', numel(id), numel(trainOpts.train), numel(trainOpts.val));

assert(min(size(m.Data(1).vbmat) == [netparams.N netparams.nwin netparams.nsigs]));

[refnet, stats] = tRefNet_train(refnet, db, @getBatch, trainOpts) ;

% inputs.data = gpuArray(inputs.data);
% res = my_simplenn(refnet,inputs.data);
% 
% matout = res(end).x;
% pout =  (matout - bmref) ./ (matout .* (1 - matout));
% 
% res = my_simplenn(refnet,inputs.data,pout);

