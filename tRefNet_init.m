function net = tRefNet_init(params)

model = params.model; % CCC_CNN | CNN | BiLSTM | MultiBranch CNN

if strcmp(model,'CCC_CNN')
    net = CCC_CNN_init(params);
end
if strcmp(model, 'CNN')
    net = CNN_init(params);
end
if strcmp(model, 'BiLSTM')
    net = BiLSTM_init(params);
end

function net = BiLSTM_init(params)
    
    marray = params.marray;
    msize = numel(marray);
    N = params.N;
    nsigs = params.nsigs;
    
    d = 96 ;  % number of hidden units
    clipGrad = 10;

    xi = Input('gpu', true) ;  % [N nwin nsigs bsize]
    % Select one random window for LSTM computation of size N
    %x = permute(xi(:,randi( size(xi,2), 1 ),:,:),[3 4 1 2]);   % [dim bsize N]
    x = permute(xi(:,1,:,:),[3 4 1 2]);   % [dim bsize N]

    % Forward LSTM NETWORK. 
    % initialize the shared parameters for an LSTM with d units
    [W, b] = vl_nnlstm_params(d, nsigs) ;

    h = cell(N, 1);
    c = cell(N, 1);
    h{1} = zeros(d, size(x,2), 'single');
    c{1} = zeros(d, size(x,2), 'single');

    % compute LSTM hidden states for all time steps
    for t = 1:N-1
    [h{t+1}, c{t+1}] = vl_nnlstm(x(:,:,t), h{t}, c{t}, W, b, 'clipGrad', clipGrad) ;
    end

    % Backward LSTM NETWORK. 
    % size(text,2) =  (the batch size).
    % initialize the shared parameters for an LSTM with d units
    [Wg, bg] = vl_nnlstm_params(d, nsigs) ;

    g = cell(N, 1);
    gb = cell(N, 1);
    g{end} = zeros(d, size(x,2), 'single');
    gb{end} = zeros(d, size(x,2), 'single');

    % compute LSTM hidden states for all time steps
    for t = N:-1:2
        [g{t-1}, gb{t-1}] = vl_nnlstm(x(:,:,t), g{t}, gb{t}, Wg, bg, 'clipGrad', clipGrad) ;
    end

    % concatenate hidden states along 3rd dimension, ignoring initial state.
    % H and G will have size [d, batchSize, N - 2]
    H = cat(3, h{2:end}) ;
    G = cat(3, g{2:end}) ;
    
    S = cat(1, H, G);
    
    % final projection applied from all timeSteps for each batchSize 
    fc1 = vl_nnconv(permute(S, [3 1 4 2]), 'size', [N - 2, 2*d, 1, N*d/2] ) ;  % permute(S) = [N-2 2*d 1 batchSize]
    fr1 = vl_nnrelu(fc1);
    fc2 = vl_nnconv(fr1, 'size',  [1, 1, N*d/2, N*d/2]) ; 
    fr2 = vl_nnrelu(fc2);
    prediction = vl_nnconv(fr2, 'size',  [1, 1, N*d/2, msize]) ; 

    % the ground truth "next" sample
    label = Input('gpu', true)  ;  % class value
    
    % compute loss and error
    loss = vl_nnloss(prediction, label, 'loss', 'softmaxlog') ;
    err = vl_nnloss(prediction, label, 'loss', 'classerror') ;
    top5err = vl_nnloss(prediction, label, 'loss', 'topkerror','TopK',5) ;

    % use workspace variables' names as the layers' names, and compile net
    Layer.workspaceNames() ;
    net = Net(loss,err,top5err) ;


function net = CCC_CNN_init(params)
    % This network description derives from the matconvnet my_simple_nn
    % framework. At the end there is a conversion to autoNN framework
    
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
                               'weights', {{1/wconvsize * ones(wconvsize,1,1,nsigs, 'single'), zeros(1,nsigs, 'single')}}, ...
                               'learningRate', [0 0], ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'relu') ;   


    net.layers{end+1} = struct('type', 'ccc', ...
                               'weights', {{ ones(N2,msize,nsigs,1, 'single')}}, ... % zeros(msize,1,nsigs, 'single')}}, ...
                               'marray', marray, ...
                               'sigma', sigma, ...
                               'forward', @nncu_ccc_forward, ...
                               'backward', @nncu_ccc_backward, ...
                               'learningRate', [0] );
    %'learningRate', [0], ...

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(1,9,1, 128, 'single'), zeros(1, 128, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'bnorm', ...
                                 'weights', {{ones(128, 1, 'single'), zeros(128, 1, 'single'), ...
                                   zeros(128, 2, 'single')}}, ...
                                 'epsilon', 1e-4 );

    net.layers{end+1} = struct('type', 'relu') ;  

    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [1 2], ...
                               'stride', [1 2], ...
                               'pad', 0) ;



    % net.layers{end+1} = struct('type', 'dropout', ...
    %                              'rate', 0.5) ;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(1,5,128, 64, 'single'), zeros(1, 64, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'bnorm', ...
                                 'weights', {{ones(64, 1, 'single'), zeros(64, 1, 'single'), ...
                                   zeros(64, 2, 'single')}}, ...
                                 'epsilon', 1e-4) ;

    net.layers{end+1} = struct('type', 'relu') ;  

    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [1 2], ...
                               'stride', [1 2], ...
                               'pad', 0) ;


    % net.layers{end+1} = struct('type', 'dropout', ...
    %                              'rate', 0.5) ;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(1,5,64, 32, 'single'), zeros(1, 32, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'bnorm', ...
                                 'weights', {{ones(32, 1, 'single'), zeros(32, 1, 'single'), ...
                                   zeros(32, 2, 'single')}}, ...
                                 'epsilon', 1e-4 );

    net.layers{end+1} = struct('type', 'relu') ;  

    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [1 2], ...
                               'stride', [1 2], ...
                               'pad', 0) ;


    % net.layers{end+1} = struct('type', 'dropout', ...
    %                              'rate', 0.5) ;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% FC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nwin = (((((nwin - 8) / 2) - 4) / 2) - 4) / 2;
    N3 = msize; % (((((N - 8) / 2) - 4) / 2) - 4) / 2;

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(N3,nwin,32, msize *  nwin , 'single'),zeros(1,msize *  nwin ,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'bnorm', ...
                                 'weights', {{ones(msize *  nwin, 1, 'single'), zeros(msize *  nwin, 1, 'single'), ...
                                   zeros(msize *  nwin, 2, 'single')}}, ...
                                 'epsilon', 1e-4 );

    net.layers{end+1} = struct('type', 'relu') ;

    net.layers{end+1} = struct('type', 'dropout', ...
                                 'rate', 0.5) ;

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(1,1,msize *  nwin , msize, 'single'),zeros(1,msize ,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'softmaxloss') ;

    % Consolidate the network, fixing any missing option
    % in the specification above
    net = vl_simplenn_tidy(net) ;
    
    net = Net(net) ; % Convert to AUTONN Framework
    


function net = CNN_init(params)
    % This network description derives from the matconvnet my_simple_nn
    % framework. At the end there is a conversion to autoNN framework
    
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(9,9,1, 128, 'single'), zeros(1, 128, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'bnorm', ...
                                 'weights', {{ones(128, 1, 'single'), zeros(128, 1, 'single'), ...
                                   zeros(128, 2, 'single')}}, ...
                                 'epsilon', 1e-4 );

    net.layers{end+1} = struct('type', 'relu') ;  

    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', [2 2], ...
                               'pad', 0) ;



    % net.layers{end+1} = struct('type', 'dropout', ...
    %                              'rate', 0.5) ;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(5,5,128, 64, 'single'), zeros(1, 64, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'bnorm', ...
                                 'weights', {{ones(64, 1, 'single'), zeros(64, 1, 'single'), ...
                                   zeros(64, 2, 'single')}}, ...
                                 'epsilon', 1e-4) ;

    net.layers{end+1} = struct('type', 'relu') ;  

    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', [2 2], ...
                               'pad', 0) ;


    % net.layers{end+1} = struct('type', 'dropout', ...
    %                              'rate', 0.5) ;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(5,5,64, 32, 'single'), zeros(1, 32, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'bnorm', ...
                                 'weights', {{ones(32, 1, 'single'), zeros(32, 1, 'single'), ...
                                   zeros(32, 2, 'single')}}, ...
                                 'epsilon', 1e-4 );

    net.layers{end+1} = struct('type', 'relu') ;  

    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', [2 2], ...
                               'pad', 0) ;


    % net.layers{end+1} = struct('type', 'dropout', ...
    %                              'rate', 0.5) ;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% FC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nwin = (((((nwin - 8) / 2) - 4) / 2) - 4) / 2;
    N3 =  (((((N - 8) / 2) - 4) / 2) - 4) / 2;

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(N3,nwin,32, msize *  nwin , 'single'),zeros(1,msize *  nwin ,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'bnorm', ...
                                 'weights', {{ones(msize *  nwin, 1, 'single'), zeros(msize *  nwin, 1, 'single'), ...
                                   zeros(msize *  nwin, 2, 'single')}}, ...
                                 'epsilon', 1e-4 );

    net.layers{end+1} = struct('type', 'relu') ;

    net.layers{end+1} = struct('type', 'dropout', ...
                                 'rate', 0.5) ;

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(1,1,msize *  nwin , msize, 'single'),zeros(1,msize ,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'softmaxloss') ;

    % Consolidate the network, fixing any missing option
    % in the specification above
    net = vl_simplenn_tidy(net) ;
    
    net = Net(net) ; % Convert to AUTONN Framework
    
