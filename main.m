% EX 5- unsupervise learning
clear all;
close all;
clc;

mpath = strrep(which(mfilename),[mfilename '.m'],'');
addpath([mpath 'ActivationFunctions']);

%% a + b. using informax ICA to separate noiseless mix
% Parameters and initialization
nDimentions=3;
nEpochs=10;
activeFunction=@Sigmoid;
W=rand(nDimentions);   %weights matrix
eta=0.1;        %learning rate

% Extract source data
for i=1:3
    filename=append(".\Data\source", num2str(i),".wav");
    [sources(:,i),Fs(i)]=audioread(filename);
end

% Extract noiseless mix data
for i=1:3
    filename=append(".\Data\mix", num2str(i),".wav");
    [noiseless(:,i),Fs(i)]=audioread(filename);
end

% Extract noisy mix data
for i=1:7
    filename=append(".\Data\noisy_mix", num2str(i),".wav");
    [noisy(:,i),Fs(i)]=audioread(filename);
end


%% ICA
separate_on = noiseless;

% separate_on = noisy(:,randperm(7,3)); %randomly choose 3 channels
sampleSize=size(separate_on,1);
for epoch=1:nEpochs
    randOrder = randperm(sampleSize);
    for i=1:sampleSize
        xi=separate_on(randOrder(i),:)';
        g=activeFunction(W*xi);
        y=1-2*g;
        dw=eta*(inv(W')+y*xi'); %learning rule of weights
        W=W+dw; %update weights
    end
    eta=0.1/(1+1e-4*epoch); %update eta each epoch
end
% Output Seperated Sources
separated=(W*separate_on')';


%% plot
channel_N = size(sources, 2);

% Compute correlation with the source and seperated data
correlation_mat = abs(corr(separated, sources));
max_corr = max(correlation_mat); %extract max correlations

% fix shifting
max_indexes = find(correlation_mat == max_corr)- [0;3;6]; 
separated = separated(:,max_indexes');

% Plot
figure
sgtitle('Corrolation between source and seperated data')
for i = 1 : channel_N
    subplot(2, channel_N, i);
    plot(sources(:,i));
    title(['Source ',num2str(i)]);
    subplot(2,channel_N ,i + channel_N);
    plot(separated(:,i));
    title(['Seperated' ,num2str(i), newline,'Cor - ' num2str(max_corr(i))]); 
end
