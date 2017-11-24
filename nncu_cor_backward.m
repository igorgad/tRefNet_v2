function resi = nncu_cor_backward  (layer,resi,reso)

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
%     
%     cmb = 1;
%     for st1 = 1:nsig
%         for st2 = st1+1:nsig
% 
%             wxm = wm(:,:,st1);
%             wym = wm(:,:,st2);
% 
%             x = reshape(resi.x(:,:,st1,:),[N nwin bsize]);
%             y = reshape(resi.x(:,:,st2,:),[N nwin bsize]);
% 
%             
%             for d = 1:bsize
% 
%                 for w = 1:nwin
% 
%                    xl = x(:,w,d);
%                    yl = y(:,w,d);
% 
%                   zm(:,w,cmb,d) = ([ xcorr(xl , yl, numel(marray)/2 - 1); 0]);
%                 end
%             end
% 
%             cmb = cmb + 1;
%         end
%     end
% 
%     zm = (zm - repmat(min(zm,[],1),[msize 1 1 1]) ) ./ ( repmat(max(zm,[],1),[msize 1 1 1]) - repmat(min(zm,[],1),[msize 1 1 1]));
% 
%     dm = pm .* zm;
% 
%     dm(:,:,2,:) = dm(:,:,1,:) * -1;  
% 
%     ddm = reshape(dm,[msize 1 nwin nsig bsize]);
%     ddm = permute(repmat(ddm,[1 N 1 1 1]), [2 1 3 4 5]);
% 
%     wwm = reshape(wm,[N msize 1 nsig 1]);
%     wwm = repmat(wwm,[1 1 nwin 1 bsize]);
% 
%     ppp = times(wwm,ddm);
% 
%     ppp = sum(ppp,2) / msize;
%     ppp = reshape(ppp,[N nwin nsig bsize]);

    resi.dzdx = zeros([N nwin nsig bsize], 'single','gpuArray');
% 
%     xx = reshape(resi.x, [N 1 nwin nsig bsize]);
%     xx = repmat(xx,[1 msize 1 1]);
% 
%     dw = times(xx,ddm);

    resi.dzdw{1} = zeros([N msize nsig],'single','gpuArray') ;
    
    clearvars -except resi
end

