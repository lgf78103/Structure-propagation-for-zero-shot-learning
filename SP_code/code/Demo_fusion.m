% We provide three versions SynC_fast_fusion for different embedding 

%% SynC_fast_fusion Demo_fusion for AWA
display('Training in cross-validation');
opt.lambda = 2 .^ (-24 : -9);
opt.Sim_scale = 2 .^ (-5 : 5);
%% ################## IMPORTANT ##################
% Note that, the above range of "hyper-parameters" is for "Demo_fusion" only.
% The range used in our experiments for AwA in SynC_fast_fusion.m is:
% opt.lambda = 2 .^ (-24 : -9);
% opt.Sim_scale = 2 .^ (-5 : 5);
% Please also check the instruction in SynC_fast_fusion.m for the setting of hyper-parameters and inputs.
%% ###############################################
cname={'OVO'};
opt.loss_type = cname{1};
opt.ind_split =[];

% SynC_fast_fusion('train', 'AWA', opt, []);
% 
% display('Collecting validation results');
% SynC_fast_fusion('val', 'AWA', opt, []);
% 
% display('Training w.r.t. to the best hyper-parameters and testing on the unseen classes');
% SynC_fast_fusion('test', 'AWA', opt, []);

% display('You can also directly train a model and test, given a pair of selected lambda and Sim_scale');
lambda =0.00001209;
Sim_scale =0.4;
SynC_fast_fusion('test', 'AWA', opt, [lambda, Sim_scale]);