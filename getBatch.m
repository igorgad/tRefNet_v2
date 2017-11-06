function [im, lb] = getBatch(vbdb, batch)
%GETBATCH  Get a batch of training data
%   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
%   and labels LABEL from IMDB according to the list of images
%   BATCH.

ref = zeros(numel(batch),1);

for b = 1:numel(batch)
   im(:,:,:,b) = vbdb.Data(batch(b)).vbmat; 
   ref(b) = vbdb.Data(batch(b)).ref + 96; 
end

lb = single(ref);
im(isnan(im)) = 0;

end

