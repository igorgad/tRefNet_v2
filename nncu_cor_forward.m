function reso = nncu_cor_forward (layer,resi,reso)

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
    
    cmb = 1;
    for st1 = 1:nsig
        for st2 = st1+1:nsig

            x = reshape(resi.x(:,:,st1,:),[N nwin bsize]);
            y = reshape(resi.x(:,:,st2,:),[N nwin bsize]);

            for d = 1:bsize

                for w = 1:nwin

                   xl = x(:,w,d)';
                   yl = y(:,w,d)';

                  zm(:,w,cmb,d) =  vl_nnconv(xl,yl,[],'Pad',[0 0 (numel(marray)/2 - 1) (numel(marray)/2)]);
                end
            end

            cmb = cmb + 1;
        end
    end

    zm = (zm - repmat(min(zm,[],1),[msize 1 1 1]) ) ./ ( repmat(std(zm,0,1),[msize 1 1 1]) );

    zm(isnan(zm)) = 0;

    reso.x = single(zm);

    clearvars -except reso
end