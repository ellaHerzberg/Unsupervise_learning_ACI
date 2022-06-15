function W = Sanger(data)
% Sanger's rule function.
% finds the weights that represent the eigenvectors for the 
% guven data

% Parameters and initialization
sampleSize = size(data,1);
nEpochs = 50;
eta = 1e-3;
W = 0.01*rand(3,7); % initialize weights randomly

% Converge max eigenvector at a time
for epoch = 1:nEpochs
    randsample=randperm(sampleSize); % Shuffle the sampels
    
    for i = 1:sampleSize
        sanger_sample = data(randsample(i),:)'; %get sample
        y = W*sanger_sample; % feed forward
        
        % Update weights
        dW = eta*(y*sanger_sample'-tril(y*y')*W);
        W = W + dW;
    end
    eta = 0.1/(1+1e-4*epoch); % update learning rate
end
W = W'; % return eigs as coloums