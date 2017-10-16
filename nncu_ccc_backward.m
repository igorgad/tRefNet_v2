function resi = nncu_ccc_backward  (layer,resi,reso)

    %%%%%%%%% PARAMS %%%%%%%%%%%%%
    marray = layer.marray;
    msize = length(marray);
    N = size(resi.x,1);
    nwin = size(resi.x,2);
    nsig = size(resi.x,3);
    bsize = size(resi.x,4);
    sigma = layer.sigma;
    
    pm = reso.dzdx;
    wm = layer.weights{1};
    
    %%%%%%%%%%%%%%%%%%%% CPU %%%%%%%%%%%%%%%%%%%%
    
    if isa(resi.x, 'single') || isa(resi.x, 'double')
    
        cmb = 1;
        for st1 = 1:nsig
            for st2 = st1+1:nsig

                wxm = wm(:,:,st1);
                wym = wm(:,:,st2);

                x = reshape(resi.x(:,:,st1,:),[nwin N bsize]);
                y = reshape(resi.x(:,:,st2,:),[nwin N bsize]);

                parfor d = 1:bsize

                    for w = 1:nwin
                       for m=1:msize

                          zm(m,w,cmb,d) = ACm_prime(x(w,:,d) .* wxm(:,m)', y(w,:,d) .* wym(:,m)', marray(m), sigma);
                       end
                    end
                end

                cmb = cmb + 1;
            end
        end
        
        
        dm = pm .* zm;

        dm(:,:,2,:) = dm(:,:,1,:) * -1;    


        ddm = reshape(dm,[msize 1 nwin nsig bsize]);
        ddm = repmat(ddm,[1 N 1 1 1]);

        wwm = reshape(wm,[msize N 1 nsig 1]);
        wwm = repmat(wwm,[1 1 nwin 1 bsize]);

        ppp = times(ddm,wwm); % THIS IS WRONG. But for now we will accept it
        psm = sum(ppp,1) / msize;

        resi.dzdx = reshape(psm,[N nwin nsig bsize]);

        ppw = mean(mean(ppp,3),5);

        resi.dzdw{1} = reshape(ppw,[N msize nsig]);
        resi.dzdw{1} = resi.dzdw{1};
    end
    
    %%%%%%%%%%%%%%%%%%%% GPU %%%%%%%%%%%%%%%%%%%%
    
     if isa(resi.x, 'gpuArray')

        acm_prime_ker = parallel.gpu.CUDAKernel('xtropy_refnet3d.ptx','xtropy_refnet3d.cu','ACm_prime');
        acm_prime_ker.GridSize = [1024 1024 64];
        acm_prime_ker.ThreadBlockSize = [16 8 8];

        cmb = 1;
        for st1 = 1:nsig
            for st2 = st1+1:nsig

                wxm = wm(:,:,st1);
                wym = wm(:,:,st2);

                x = reshape(resi.x(:,:,st1,:),[nwin N bsize]);
                y = reshape(resi.x(:,:,st2,:),[nwin N bsize]);

                gpu_inx = gpuArray(single(x));
                gpu_iny = gpuArray(single(y));
                gpu_wx  = gpuArray(single(wxm));
                gpu_wy  = gpuArray(single(wym));
                gpu_acm  = zeros(nwin,msize,bsize,'single','gpuArray');
                gpu_m   = gpuArray(int32(marray));

                k = feval(acm_prime_ker, gpu_acm, gpu_inx, gpu_iny, gpu_wx, gpu_wy, gpu_m, single(sigma), uint32(msize), uint32(N), uint32(nwin), uint32(bsize));

                %kout = gather(k);

                zm(:,:,cmb,:) = reshape(k,[msize nwin 1 bsize]);
            end
        end

        dm = pm .* zm;

        dm(:,:,2,:) = dm(:,:,1,:) * -1;    


        ddm = reshape(dm,[msize 1 nwin nsig bsize]);
        ddm = repmat(ddm,[1 N 1 1 1]);

        wwm = reshape(wm,[msize N 1 nsig 1]);
        wwm = repmat(wwm,[1 1 nwin 1 bsize]);

        ppp = times(ddm,wwm); % THIS IS WRONG. But for now we will accept it
        psm = sum(ppp,1) / msize;

        resi.dzdx = reshape(psm,[N nwin nsig bsize]);

        ppw = mean(mean(ppp,3),5);

        resi.dzdw{1} = reshape(ppw,[N msize nsig]);
        %resi.dzdw{1} = single(gather(resi.dzdw{1}));

     end
    
end