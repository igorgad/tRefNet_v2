function [im, lb] = getBatch(vbdb, batch)
%GETBATCH  Get a batch of training data
%   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
%   and labels LABEL from IMDB according to the list of images
%   BATCH.

ref = reshape(vbdb.ref(:,:,:,batch), [1 numel(batch)]);

im = single(vbdb.data(:,:,:,batch));
lb = ref + 96;

