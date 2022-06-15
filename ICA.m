function [separated,W] = ICA(data)
% The function calculate infomax ICA for the data

% Parameters and initialization
nDimentions=3;
nEpochs=10;
activeFunction=@Sigmoid;
W=rand(nDimentions);   %weights matrix
eta=0.1;        %learning rate


sampleSize=size(data,1);
for epoch=1:nEpochs
    randOrder = randperm(sampleSize);
    for i=1:sampleSize
        xi=data(randOrder(i),:)';
        g=activeFunction(W*xi);
        y=1-2*g;
        dw=eta*(inv(W')+y*xi'); %learning rule of weights
        W=W+dw; %update weights
    end
    eta=0.1/(1+1e-4*epoch); %update eta each epoch
end
separated=(W*data')';
end

