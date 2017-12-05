 clear;

rng('default');

run(fullfile(matlabroot, 'toolbox/autonn/setup_autonn.m')) ;  % add AutoNN to the path

%%% INITIALIZE CNN
netparams.nsigs     = 2;
netparams.N         = 128;
netparams.wconvsize = 8;
netparams.marray    = -80:79;
%netparams.marray    = -floor(2*(netparams.N-netparams.wconvsize+1)/3):floor(2*(netparams.N-netparams.wconvsize+1)/3 -1 );
netparams.sigma     = 0.1;
netparams.nwin      = 64;
netparams.f         =  0.02;
netparams.batch_size = 256;
netparams.model = 'BiLSTM';

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

id = int32(find(combClass == 4));
trainOpts.train = id(randi(numel(id)-1,numel(id)*0.8*10,1));
id2 = setdiff(id,trainOpts.train);
trainOpts.val = id2(randi(numel(id2)-1,numel(id)*0.2*10,1));

prefix = 'REFTEST_BiLSTM';
%prefix = 'AUTOTEST4_WL1';

% Train
%trainOpts.gpus = [] ;
trainOpts.gpus = [1] ;
trainOpts.batchSize = netparams.batch_size ;
trainOpts.plotDiagnostics = false ;
trainOpts.plotStatistics = true;
trainOpts.numEpochs = 200 ;
trainOpts.numSubBatches = 1 ;
trainOpts.learningRate = 0.01; % 1./( 50 + exp( 0.05 * (1:trainOpts.numEpochs) ) ); % [0.01 * ones(1,25), 0.007 * ones(1,100), 0.004 * ones(1,200), 0.002 * ones(1,500)] ;
trainOpts.momentum = 0.8 ;
trainOpts.weightDecay = 0.00 ;
%trainOpts.solver = @adam;        % Use ADAM solver instead of SGD
trainOpts.continue = false;                                                                                                                                                                                   
trainOpts.expDir = sprintf('/media/pepeu/582D8A263EED4072/DATASETS/MedleyDB/mat_conv_data/%s_NT%d_NV%d_bsize%d_N%d_NW%d', prefix, numel(trainOpts.train), numel(trainOpts.val), netparams.batch_size, netparams.N, netparams.nwin)

fprintf ('Total available inputs %d. Train %d | Eval %d\n', numel(id), numel(trainOpts.train), numel(trainOpts.val));

assert(min(size(m.Data(1).vbmat) == [netparams.N netparams.nwin netparams.nsigs]));


info = tRefNet_train(refnet, m, @getBatch, trainOpts) ;

