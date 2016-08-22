function [ output_args ] = InitPCNN( ~ )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% 
% ENABLE GPU support (in my_build.m) and MATLAB Parallel Pool to speed up computation (parpool) 

if ~isdeployed
    addpath('brox_OF'); % Brox 2004 optical flow
end
matconvpath = 'matconvnet-1.0-beta11'; % MatConvNet
run([matconvpath '/my_build.m']); % compile: modify this file to enable GPU support (much faster)
run([matconvpath '/matlab/vl_setupnn.m']) ; % setup  

end

