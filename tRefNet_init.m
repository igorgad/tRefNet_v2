function net = tRefNet_init(params)


f = params.f;
marray = params.marray;
msize = numel(marray);
N = params.N;
N2 = params.N - params.wconvsize + 1;
nsigs = params.nsigs;
wconvsize = params.wconvsize;
nwin = params.nwin;
bsize = params.batch_size;
sigma = params.sigma;



net.meta.inputSize = [N nwin nsigs bsize] ;

net.layers = {} ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.1 * randn(wconvsize,1,1,nsigs, 'single'), zeros(1,nsigs, 'single')}}, ...
                           'learningRate', [10 1], ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;   
                       
                       
net.layers{end+1} = struct('type', 'ccc', ...
                           'weights', {{ 0.1 * randn(N2,msize,nsigs,1, 'single')}}, ... % zeros(msize,1,nsigs, 'single')}}, ...
                           'marray', marray, ...
                           'sigma', sigma, ...
                           'forward', @nncu_ccc_forward, ...
                           'backward', @nncu_ccc_backward, ...
                           'learningRate', [10], ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'dropout', ...
                             'rate', 0.7) ;

%'learningRate', [0], ...

%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,9,1, 128, 'single'), zeros(1, 128, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [1 2], ...
                           'stride', [1 2], ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;     

net.layers{end+1} = struct('type', 'dropout', ...
                             'rate', 0.9) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,5,128, 64, 'single'), zeros(1, 64, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [1 2], ...
                           'stride', [1 2], ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'dropout', ...
                             'rate', 0.75) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,5,64, 32, 'single'), zeros(1, 32, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [1 2], ...
                           'stride', [1 2], ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;  

net.layers{end+1} = struct('type', 'dropout', ...
                             'rate', 0.75) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%% FC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nwin = (((((nwin - 8) / 2) - 4) / 2) - 4) / 2;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(msize,nwin,32, msize *  nwin , 'single'),zeros(1,msize *  nwin ,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'dropout', ...
                             'rate', 0.5) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,msize *  nwin , msize, 'single'),zeros(1,msize ,'single')}}, ...
                           'stride', 1, ...
                           'learningRate', [0.1 1], ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'dropout', ...
                             'rate', 0.5) ;

net.layers{end+1} = struct('type', 'softmaxloss') ;

% Consolidate the network, fixing any missing option
% in the specification above


net = vl_simplenn_tidy(net) ;

