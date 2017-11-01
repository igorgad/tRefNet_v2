function [im, lb] = getBatch(vbdb, batch)
%GETBATCH  Get a batch of training data
%   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
%   and labels LABEL from IMDB according to the list of images
%   BATCH.


for b = 1:numel(batch);
    c = floor(batch/vbdb.chunkSize) + 1;
    
    chunk = vbdb.(sprintf('chunk_%d',c(b)));
    
    im(:,:,1,b) = single(chunk.(sprintf('ac_%d',batch(b))).vb1);
    im(:,:,2,b) = single(chunk.(sprintf('ac_%d',batch(b))).vb2);
    
    ref(b) = chunk.(sprintf('ac_%d',batch(b))).ref;
end

lb = ref + 96;

im(isnan(im)) = 0;

