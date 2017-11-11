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

                x = reshape(resi.x(:,:,st1,:),[N nwin bsize]);
                y = reshape(resi.x(:,:,st2,:),[N nwin bsize]);

                parfor d = 1:bsize

                    for w = 1:nwin
                        
                       xl = x(:,w,d);
                       yl = y(:,w,d);
                        
                       for m=1:msize

                          zm(m,w,cmb,d) = ACm_prime(xl .* wxm(:,m), yl .* wym(:,m), marray(m), sigma);
                       end
                    end
                end

                cmb = cmb + 1;
            end
        end
        
        %zm = (zm - repmat(min(zm,[],1),[msize 1 1 1]) ) ./ ( repmat(max(zm,[],1),[msize 1 1 1]) - repmat(min(zm,[],1),[msize 1 1 1]));
        
        dm = pm .* zm;

        dm(:,:,2,:) = dm(:,:,1,:) * -1;  
        
        ddm = reshape(dm,[msize 1 nwin nsig bsize]);
        ddm = permute(repmat(ddm,[1 N 1 1 1]), [2 1 3 4 5]);
        
        wwm = reshape(wm,[N msize 1 nsig 1]);
        wwm = repmat(wwm,[1 1 nwin 1 bsize]);
        
        ppp = times(wwm,ddm);
        
        ppp = sum(ppp,2) / msize;
        ppp = reshape(ppp,[N nwin nsig bsize]);
        
        resi.dzdx = ppp;
      
        xx = reshape(resi.x, [N 1 nwin nsig bsize]);
        xx = repmat(xx,[1 msize 1 1]);
        
        dw = times(xx,ddm);

        resi.dzdw{1} = reshape(mean(mean(dw,3),5),[N msize nsig]) ;
    end
    
    %%%%%%%%%%%%%%%%%%%% GPU %%%%%%%%%%%%%%%%%%%%
    
     if isa(resi.x, 'gpuArray')

        acm_prime_ker = parallel.gpu.CUDAKernel('cuda/xtropy_refnet3d.ptx','cuda/xtropy_refnet3d.cu','ACm_prime');
        acm_prime_ker.GridSize = [msize nwin min(64,bsize)];
        acm_prime_ker.ThreadBlockSize = [4 4 4];

        cmb = 1;
        for st1 = 1:nsig
            for st2 = st1+1:nsig

                wxm = wm(:,:,st1)';
                wym = wm(:,:,st2)';

                x = reshape(resi.x(:,:,st1,:),[N nwin bsize]);
                y = reshape(resi.x(:,:,st2,:),[N nwin bsize]);

                gpu_inx = gpuArray(single(x));
                gpu_iny = gpuArray(single(y));
                gpu_wx  = gpuArray(single(wxm));
                gpu_wy  = gpuArray(single(wym));
                gpu_acm  = zeros(1,msize*nwin*bsize,'single','gpuArray');
                gpu_m   = gpuArray(int32(marray));
                
                gpu_inx = reshape(gpu_inx,1,[]);
                gpu_iny = reshape(gpu_iny,1,[]);
                
                gpu_wx = reshape(gpu_wx.',1,[]);
                gpu_wy = reshape(gpu_wy.',1,[]);

                k = feval(acm_prime_ker, gpu_acm, gpu_inx, gpu_iny, gpu_wx, gpu_wy, gpu_m, single(sigma), uint32(msize), uint32(N), uint32(nwin), uint32(bsize));

                %kout = gather(k);

                zm(:,:,cmb,:) = reshape(k,[msize nwin 1 bsize]);
            end
        end
        
        zm = (zm - repmat(mean(zm,1),[msize 1 1 1]) ) ./ ( 4 * repmat(std(zm,0,1),[msize 1 1 1]) );
        zm(isnan(zm)) = 0;
        
        dm = times(pm,zm);
        dm(:,:,2,:) = dm(:,:,1,:) * -1;  
        
        % Compute dzdx

        dm = reshape(dm,[msize 1 nwin nsig bsize]);         
        wm = reshape(wm,[N msize 1 nsig 1]);

        resi.dzdx = pagefun(@mtimes,wm,dm .* 1/msize);        
        resi.dzdx = reshape(resi.dzdx,[N nwin nsig bsize]);
        
        % Compute dzdw
        
        xy = mean(mean(resi.x,2),4);
        xy = reshape(xy, [N 1 nsig]);
        xy = repmat(xy,[1 msize 1]);
        
        dm = mean(mean(dm,3),5);
        dm = reshape(dm,[1 msize nsig]);
        dm = repmat(dm,[N 1 1]);

        resi.dzdw{1} = pagefun(@times,xy,dm);
     end
     
    clearvars -except resi
end

