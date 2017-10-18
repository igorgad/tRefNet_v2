clear all;

medMatfilename = '/media/pepeu/582D8A263EED4072/MedleyDB/autotest_medleyVBRdataset.mat'

rng('default');

%%% INITIALIZE CNN
netparams.nsigs     = 2;
netparams.N         = 146;
netparams.wconvsize = 3;
%netparams.marray    = -256:255;
netparams.marray    = -floor(2*(netparams.N-netparams.wconvsize+1)/3):floor(2*(netparams.N-netparams.wconvsize+1)/3 -1 );
netparams.sigma     = 1;
netparams.nwin      = 32;
netparams.batch_size = 32;
netparams.f         =  0.01;

refnet = tRefNet_init(netparams);
%my_simplenn_display(refnet);

N = netparams.N;
nwin = netparams.nwin;
batch_size = netparams.batch_size;

load(medMatfilename);

fprintf ('Total available inputs %d\n', size(vbdb.data,4));


% Train
trainOpts.expDir = '/media/pepeu/582D8A263EED4072/MedleyDB/mat_conv_data/autotest' ;
%trainOpts.gpus = [] ;
trainOpts.gpus = [1] ;
trainOpts.batchSize = netparams.batch_size ;
trainOpts.plotDiagnostics = true ;
trainOpts.plotStatistics = true;
trainOpts.numEpochs = 300 ;
trainOpts.epochSize = inf ;
trainOpts.numSubBatches = 1 ;
trainOpts.learningRate = 0.02 ;
trainOpts.momentum = 0.9 ;
trainOpts.weightDecay = 0.0005 ;
trainOpts.errorFunction = 'multiclass' ;
trainOpts.train = randi(numel(find(vbdb.set == 4)),512,1);
trainOpts.val = randi(numel(find(vbdb.set == 4)),96,1);
trainOpts.prefetch = false;
trainOpts.continue = false;
trainOpts.profile = false;

db = vbdb;

assert(min(size(db.data(:,:,:,1)) == [netparams.N netparams.nwin netparams.nsigs]));

[refnet, stats] = tRefNet_train(refnet, db, @getBatch, trainOpts) ;


% inputs.data = gpuArray(inputs.data);
% res = my_simplenn(refnet,inputs.data);
% 
% matout = res(end).x;
% pout =  (matout - bmref) ./ (matout .* (1 - matout));
% 
% res = my_simplenn(refnet,inputs.data,pout);

