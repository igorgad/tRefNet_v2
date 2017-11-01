function [im, lb] = getBatch(vbdb, batch)
%GETBATCH  Get a batch of training data
%   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
%   and labels LABEL from IMDB according to the list of images
%   BATCH.

im = vbdb.data(:,:,:,batch);
ref = reshape(vbdb.ref(:,:,:,batch),[1 numel(batch)]);

lb = ref + 96;

im(isnan(im)) = 0;

