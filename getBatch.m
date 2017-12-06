function [im, lb] = getBatch(vbdb, batch)
%GETBATCH  Get a batch of training data
%   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
%   and labels LABEL from IMDB according to the list of images
%   BATCH.



ref = zeros(numel(batch),1);

for b = 1:numel(batch)
   im(:,:,:,b) = vbdb.m.Data(batch(b)).vbmat; 
   ref(b) = vbdb.m.Data(batch(b)).ref + 81; 
   
   lb.instComb{b} = [vbdb.allcomb.(sprintf('C_%02d', batch(b))).inst1, ' x ', vbdb.allcomb.(sprintf('C_%02d', batch(b))).inst2];
   lb.typeComb{b} = [vbdb.allcomb.(sprintf('C_%02d', batch(b))).type1, ' x ', vbdb.allcomb.(sprintf('C_%02d', batch(b))).type2];
end



lb.ref = single(ref);
im(isnan(im)) = 0;

%im(:,:,2,:) = im(:,:,2,:) .* -1;

end

