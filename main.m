% EX 5- unsupervise learning
clear all;
close all;
clc;

% Extract source data
for i=1:3
    filename=append(".\Data\source", num2str(i),".wav");
    [sources(:,i),Fs_source(i)]=audioread(filename);
end

% Extract noiseless mix data
for i=1:3
    filename=append(".\Data\mix", num2str(i),".wav");
    [noiseless(:,i),Fs_noiseless(i)]=audioread(filename);
end

% Extract noisy mix data
for i=1:7
    filename=append(".\Data\noisy_mix", num2str(i),".wav");
    [noisy(:,i),Fs_noisy(i)]=audioread(filename);
end


%%  a. + b. using informax ICA to separate noiseless mix
% randomly choose 3 channels for noisy mix
noisy_to_separated = noisy(:,randperm(7,3)); 

% Seperated Sources for noisy and noiseless with ICA
[noiseless_separated, ~] =ICA(noiseless);
[noisy_separated, W_noisy] =ICA(noisy_to_separated);

% save data
% save noiseless
for i=1:3
    filename = append('./Results/noiseless_result_a',num2str(i),'.wav');
    audiowrite(filename,rescale(noiseless_separated(:,i),-1,1),Fs_noiseless(i));
end

% save noisy
for i=1:3
    filename = append('./Results/noisy_result_b',num2str(i),'.wav');
    audiowrite(filename,rescale(noisy_separated(:,i),-1,1),Fs_noisy(i));
end

% plot
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
    disp(main_title);
    disp(corr(separated, sources));
    
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
Sanger_eigenvectors = Sanger(noisy);
% Compute eigenvectors using matlab's PCA function
pca_eigenvectors = pca(noisy,'NumComponents',3);

% Checking the distance between the PCA and the Sanger rule
for i = 1:size(pca_eigenvectors,2)
    dis_pca_sanger = abs(pca_eigenvectors)-abs(Sanger_eigenvectors);
    norm_pca_sanger(i) = norm(dis_pca_sanger(:,i));
end

disp('the distance between the PCA and the Sanger rule:');
disp(norm_pca_sanger);
disp(newline)

% Reduce the noisy data to 3 dimensions using sanger's weights
noisy_lower_dim = noisy*Sanger_eigenvectors;

% Seperated Sources for with ICA
[noisy_lower_dim_separated, W_noisy_lower_dim] =...
    ICA(noisy_lower_dim);

% save noisy loewr dimantion
for i=1:3
    filename = append('./Results/noisy_lower_dim_result_c',num2str(i),'.wav');
    audiowrite(filename,rescale(noisy_separated(:,i),-1,1),Fs_noisy(i));
end

% plot
% Compute correlation with the source and seperated data
separated = noisy_lower_dim_separated;
correlation_mat = abs(corr(separated, sources));
max_corr = max(correlation_mat); %extract max correlations

% fix shifting
max_indexes = find(correlation_mat == max_corr)- [0;3;6]; 
separated = separated(:,max_indexes');

% Plot
figure
main_title = 'Corrolation between source and seperated lower dim noisy data';

disp(main_title);
disp(corr(separated, sources));

sgtitle(main_title)
for j = 1 : channel_N
    subplot(2, channel_N, j);
    plot(sources(:,j));
    title(['Source ',num2str(j)]);
    subplot(2,channel_N ,j + channel_N);
    plot(separated(:,j));
    title(['Seperated' ,num2str(j), newline,'Cor - ' num2str(max_corr(j))]); 
end