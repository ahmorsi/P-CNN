function [ output_args ] = ComputePCNN( datadir,splitsMat,className,splitNum,compute_kernel)
%COMPUTEPCNN Summary of this function goes here
%   Detailed explanation goes here

%% P-CNN computation
% ----- PARAMETERS --------
param=[];
param.lhandposition=13; % pose joints positions in the structure (JHMDB pose format)
param.rhandposition=12;
param.upbodypositions=[1 2 3 4 5 6 7 8 9 12 13];
param.lside = 40 ; % length of part box side (also depends on the human scale)
param.savedir = sprintf('%s/cnnfeatures/%s/p-cnn_features_split%d',datadir,className,splitNum); % P-CNN results directory
param.impath = sprintf('%s/images/%s',datadir,className) ; % input images path (one folder per video)
param.imext = '.png' ; % input image extension type
param.jointpath = sprintf('%s/joint_positions/%s',datadir,className); % human pose (one folder per video in which there is a file called 'joint_positions.mat')
param.trainsplitpath = sprintf('%s/splits/%s/train%d.txt',datadir,className,splitNum); % split paths
param.testsplitpath = sprintf('%s/splits/%s/test%d.txt',datadir,className,splitNum);
param.cachepath = sprintf('%s/GPU/cache',datadir); % cache folder path
param.net_app  = load('models/imagenet-vgg-f.mat') ; % appearance net path
param.net_flow = load('models/flow_net.mat') ; % flow net path
param.batchsize = 128 ; % size of CNN batches
param.use_gpu = false ; % use GPU or CPUs to run CNN?
param.nbthreads_netinput_loading = 20 ; % nb of threads used to load input images
param.compute_kernel = compute_kernel ; % compute linear kernel and save it. If false, save raw features instead.

% get video names

splitData = splitsMat(splitNum,:);
video_names = [];
for cellidx = 1:length(splitData)
    splitCell = splitData(cellidx);
    names = splitCell{1};
    for lineIdx=1:length(names)
        line = strsplit(names{lineIdx},'/');
        if strcmp(className,line{1}) == 1
            video_names = [video_names;line(2)];
        end   
    end    
end    
% video_names = dir(param.impath);
% video_names={video_names.name};
% video_names=video_names(~ismember(video_names,{'.','..'}));

if ~exist(param.cachepath,'dir'); mkdir(param.cachepath) ; end % create cache folder

% 1 - pre-compute OF images for all videosS
compute_OF(video_names,param); % compute optical flow between adjacent frames
% 
% % 2 - extract part patches
 extract_cnn_patches(video_names,param)
% 
% %3 - extract CNN features for each patch and group per video
  extract_cnn_features(video_names,param)

%4 - compute final P-CNN features + kernels
compute_pcnn_features(param); % compute P-CNN for splitNum

end

