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
                           'weights', {{1/3 * ones(wconvsize,1,1,2, 'single'), zeros(1,2, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'ccc', ...
                           'weights', {{ones(N2,msize,nsigs,1, 'single')}}, ... % zeros(msize,1,nsigs, 'single')}}, ...
                           'marray', marray, ...
                           'sigma', sigma, ...
                           'forward', @nncu_ccc_forward, ...
                           'backward', @nncu_ccc_backward, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       

% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [2 2], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(7,7,20,50, 'single'),zeros(1,50,'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [2 2], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(20,4,50,500, 'single'),  zeros(1,500,'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(1,1,500,96, 'single'), zeros(1,96,'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'softmaxloss') ;



net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(msize,nwin,1, msize *  nwin, 'single'), zeros(1, msize *  nwin, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;                
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,msize *  nwin, msize *  nwin * 0.5, 'single'),zeros(1,msize *  nwin * 0.5,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,msize *  nwin * 0.5, msize *  nwin * 0.25, 'single'),zeros(1,msize *  nwin * 0.25,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,msize *  nwin * 0.25, msize *  nwin * 0.125 , 'single'),  zeros(1,msize *  nwin * 0.125 ,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'relu') ;
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,msize *  nwin * 0.125, msize *  nwin * 0.0625, 'single'),  zeros(1,msize *  nwin * 0.0625,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'relu') ;
                                              
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,msize *  nwin * 0.0625, msize , 'single'),  zeros(1,msize ,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
%net.layers{end+1} = struct('type', 'sigmoid') ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Consolidate the network, fixing any missing option
% in the specification above


net = vl_simplenn_tidy(net) ;

