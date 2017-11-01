 clear;

%medMatfilename = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/REFTEST_xpan10_N151_NW32_CC37860_medleyVBRdataset.mat'
%medMatfilename = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/AUTOTEST_xpan10_N151_NW32_CC37860_medleyVBRdataset.mat'
%medMatfilename = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/perfect_autotest_medleyVBRdataset.mat'

rng('default');
vl_setupnn;

%%% INITIALIZE CNN
netparams.nsigs     = 2;
netparams.N         = 151;
netparams.wconvsize = 8;
%netparams.marray    = -96:95;
netparams.marray    = -floor(2*(netparams.N-netparams.wconvsize+1)/3):floor(2*(netparams.N-netparams.wconvsize+1)/3 -1 );
netparams.sigma     = 1;
netparams.nwin      = 64;
netparams.batch_size = 128;
netparams.f         =  0.01;

medMatfilename = sprintf('/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/AUTOTEST_N%d_NW%d_XPAN10_medleyVBRdataset.mat',netparams.N,netparams.nwin);

refnet = tRefNet_init(netparams);
%my_simplenn_display(refnet);

N = netparams.N;
nwin = netparams.nwin;
batch_size = netparams.batch_size;

load(medMatfilename);
%vbdb.data(isnan(vbdb.data(:))) = 0;
%vbdb = matfile(medMatfilename);

id = find(vbdb.set == 4);

% Train
trainOpts.expDir = '/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/mat_conv_data/AUTOTEST_NT2048_NV1024_bsize128_N151_NW32_sigma1_CCCl_CNN_FC' ;
%trainOpts.gpus = [] ;
trainOpts.gpus = [1] ;
trainOpts.batchSize = netparams.batch_size ;
trainOpts.plotDiagnostics = true ;
trainOpts.plotStatistics = true;
trainOpts.numEpochs = 500 ;
trainOpts.epochSize = inf ;
trainOpts.numSubBatches = 1 ;
trainOpts.learningRate = 0.02 ;
trainOpts.momentum = 0.9 ;
trainOpts.weightDecay = 0.01 ;
trainOpts.errorFunction = 'multiclass' ;
trainOpts.train = id(randi(numel(id),2048,1));
trainOpts.val = id(randi(numel(id),1024,1));
trainOpts.prefetch = false;
trainOpts.continue = false;
trainOpts.profile = false;


fprintf ('Total available inputs. Train %d | Eval %d\n', numel(trainOpts.train), numel(trainOpts.val));

assert(min(size(vbdb.data(:,:,:,1)) == [netparams.N netparams.nwin netparams.nsigs]));

[refnet, stats] = tRefNet_train(refnet, vbdb, @getBatch, trainOpts) ;

% inputs.data = gpuArray(inputs.data);
% res = my_simplenn(refnet,inputs.data);
% 
% matout = res(end).x;
% pout =  (matout - bmref) ./ (matout .* (1 - matout));
% 
% res = my_simplenn(refnet,inputs.data,pout);

