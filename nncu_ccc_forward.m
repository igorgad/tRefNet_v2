function reso = nncu_ccc_forward (layer,resi,reso)

    %%%%%%%%% PARAMS %%%%%%%%%%%%%
    marray = layer.marray;
    msize = length(marray);
    N = size(resi.x,1);
    nwin = size(resi.x,2);
    nsig = size(resi.x,3);
    bsize = size(resi.x,4);
    sigma = layer.sigma;

    %%%%%%%%%%%%%%%%%%%% GPU %%%%%%%%%%%%%%%%%%%%

    acm_ker = parallel.gpu.CUDAKernel('xtropy_refnet3d.ptx','xtropy_refnet3d.cu','ACm');
    acm_ker.GridSize = [1024 1024 64];
    acm_ker.ThreadBlockSize = [16 8 8];
    
    wm = layer.weights{1};

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

            k = feval(acm_ker, gpu_acm, gpu_inx, gpu_iny, gpu_wx, gpu_wy, gpu_m, single(sigma), uint32(msize), uint32(N), uint32(nwin), uint32(bsize));

            %kout = gather(k);

            xgpu(:,:,cmb,:) = reshape(k,[msize nwin 1 bsize]);
          
            cmb = cmb + 1;
        end
    end
    
    xgpu = (xgpu - repmat(min(xgpu,[],1),[msize 1 1 1]) ) ./ ( repmat(max(xgpu,[],1),[msize 1 1 1]) - repmat(min(xgpu,[],1),[msize 1 1 1]));
    
    if isa(resi.x, 'single')
        reso.x = single(gather(xgpu));
    end
    if isa(resi.x, 'double')
        reso.x = double(gather(xgpu));
    end
    if isa(resi.x, 'gpuArray')
        reso.x = xgpu;
    end
    
end