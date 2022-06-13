% EX 5- unsupervise learning
clear all;
close all;
clc;

mpath = strrep(which(mfilename),[mfilename '.m'],'');
addpath([mpath 'ActivationFunctions']);

%% a. + b. using informax ICA to separate noiseless mix
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
%randomly choose 3 channels for noisy mix
noisy_to_separated = noisy(:,randperm(7,3)); 

% Seperated Sources for noisy and noiseless with ICA
[noiseless_separated, ~] =ICA(noiseless, W, nEpochs, eta, activeFunction);
[noisy_separated, W_noisy] =ICA(noisy_to_separated, W, nEpochs, eta, activeFunction);


%% plot
separated_mixes{1} = noiseless_separated;
separated_mixes{2} = noisy_separated;

channel_N = size(sources, 2);

for i = 1:2
    separated = separated_mixes{i};
    % Compute correlation with the source and seperated data
    correlation_mat = abs(corr(separated, sources));
    max_corr = max(correlation_mat); %extract max correlations

    % fix shifting
    max_indexes = find(correlation_mat == max_corr)- [0;3;6]; 
    separated = separated(:,max_indexes');

    % Plot
    figure
    main_title = 'Corrolation between source and seperated noiseless data';
    if i==2
        main_title = 'Corrolation between source and seperated noisy data';
    end
    
    sgtitle(main_title)
    for j = 1 : channel_N
        subplot(2, channel_N, j);
        plot(sources(:,j));
        title(['Source ',num2str(j)]);
        subplot(2,channel_N ,j + channel_N);
        plot(separated(:,j));
        title(['Seperated' ,num2str(j), newline,'Cor - ' num2str(max_corr(j))]); 
    end
end

%% c. Sanger's rule

% Compute eigenvectors using Sanger's rule
Sanger_eigenvectors = Sanger(noisy, eta);
% Compute eigenvectors using matlab's PCA function
pca_eigenvectors = pca(noisy);
% Compare the distance between both sets of eigenvectors
distance = sqrt(sum((Sanger_eigenvectors-pca_eigenvectors).^2));

% Reduce the noisy data to 3 dimensions using sanger's weights
noisy_lower_dim = noisy*Sanger_eigenvectors(:,1:nDimentions);

% Seperated Sources for with ICA
[noisy_lower_dim_separated, W_noisy_lower_dim] =...
    ICA(noisy_lower_dim, W, nEpochs, eta, activeFunction);

%% plot
% Compute correlation with the source and seperated data
separated = noisy_lower_dim_separated;
correlation_mat = abs(corr(separated, sources));
max_corr = max(correlation_mat); %extract max correlations

% fix shifting
max_indexes = find(correlation_mat == max_corr)- [0;3;6]; 
separated = separated(:,max_indexes');

% Plot
figure
main_title = 'Corrolation between source and seperated lower dim data';

sgtitle(main_title)
for j = 1 : channel_N
    subplot(2, channel_N, j);
    plot(sources(:,j));
    title(['Source ',num2str(j)]);
    subplot(2,channel_N ,j + channel_N);
    plot(separated(:,j));
    title(['Seperated' ,num2str(j), newline,'Cor - ' num2str(max_corr(j))]); 
end