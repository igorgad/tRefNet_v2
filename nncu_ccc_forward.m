function reso = nncu_ccc_forward (layer,resi,reso)

    %%%%%%%%% PARAMS %%%%%%%%%%%%%
    marray = layer.marray;
    msize = length(marray);
    N = size(resi.x,1);
    nwin = size(resi.x,2);
    nsig = size(resi.x,3);
    bsize = size(resi.x,4);
    sigma = layer.sigma;
    
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

                          zm(m,w,cmb,d) = ACm(xl .* wxm(:,m), yl .* wym(:,m), marray(m), sigma);
                       end
                    end
                end

                cmb = cmb + 1;
            end
        end
        
        zm = (zm - repmat(min(zm,[],1),[msize 1 1 1]) ) ./ ( repmat(std(zm,0,1),[msize 1 1 1]) );
        
        zm(isnan(zm)) = 0;
         
        reso.x = single(zm);
    end
    

    %%%%%%%%%%%%%%%%%%%% GPU %%%%%%%%%%%%%%%%%%%%

    if isa(resi.x, 'gpuArray')
        
        %mexcuda -v  ./cuda/xtropy_refnet3d_SM.cu  -dynamic '-L/usr/local/cuda/lib64/' -lcudadevrt -lcublas_device
        acm_ker = parallel.gpu.CUDAKernel('cuda/xtropy_refnet3d.ptx','cuda/xtropy_refnet3d.cu','ACm');
        acm_ker.GridSize = [msize nwin min(64,bsize)];
        acm_ker.ThreadBlockSize = [4 4 4];

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
    
                k = feval(acm_ker, gpu_acm, gpu_inx, gpu_iny, gpu_wx, gpu_wy, gpu_m, single(sigma), uint32(msize), uint32(N), uint32(nwin), uint32(bsize));

                xgpu(:,:,cmb,:) = reshape(k,[msize nwin 1 bsize]);

                cmb = cmb + 1;
            end
        end

        xgpu = (xgpu - repmat(mean(xgpu,1),[msize 1 1 1]) ) ./ ( 2 * repmat(std(xgpu,0,1),[msize 1 1 1]) );
        xgpu(isnan(xgpu)) = 0;
       
        reso.x = xgpu;
    end
    
    clearvars -except reso
end