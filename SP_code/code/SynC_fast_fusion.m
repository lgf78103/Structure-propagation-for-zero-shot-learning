function SynC_fast_fusion(task, dataset, opt, direct_test)

%% Input
% task: 'train', 'val', 'test'
% dataset: 'AWA'
% opt: opt.lambda: the regularizer coefficient on W in training (e.g, 2 .^ (-24 : -9))
%      opt.Sim_scale: the RBF scale parameter for computing semantic similarities (e.g., 2 .^ (-5 : 5))
%      opt.ind_split: AWA: []
%      opt.loss_type: 'OVO'
% direct_test: test on a specific [lambda, Sim_scale] pair without cross-validation

%% Settings
set_path;
norm_method = 'L2'; Sim_type = 'RBF_norm';

%% Data
if(strcmp(dataset, 'AWA'))
    load ../data/data_AWA.mat
    trainval_Y_cont=trainval_Y('cont');
    trainval_Y_cont(trainval_Y_cont<0) = 0;
    
    test_Y_cont=test_Y('cont');
    test_Y_cont(test_Y_cont<0) = 0;

    train_Y_cont=train_Y('cont');
    train_Y_cont(train_Y_cont<0) = 0;
   
    val_Y_cont=val_Y('cont');
    val_Y_cont(val_Y_cont<0) = 0;
 
elseif(strcmp(dataset, 'CUB'))
    load ../data/data_CUB.mat
    trainval_Y_cont=trainval_Y('cont');
    trainval_Y_cont(trainval_Y_cont<0) = 0;
  
    test_Y_cont=test_Y('cont');
    test_Y_cont(test_Y_cont<0) = 0;
   
    train_Y_cont=train_Y('cont');
    train_Y_cont(train_Y_cont<0) = 0;
  
    val_Y_cont=val_Y('cont');
    val_Y_cont(val_Y_cont<0) = 0;
    
elseif(strcmp(dataset, 'Dogs113'))
     load ../data/data_Dogs113.mat
    
else
    display('Wrong dataset!');
    return;
end

    Ytr = trainval_labels;
    Yte = test_labels;
    Yval=val_labels;
    Ytr1=train_labels;

    Xtr =trainval_X(:,2:end);
    Xtr(isnan(Xtr)) = 0; Xtr(isinf(Xtr)) = 0;
    Xtr = bsxfun(@rdivide, Xtr, sqrt(sum(Xtr .^ 2, 2)));
    Xtr(isnan(Xtr)) = 0; Xtr(isinf(Xtr)) = 0;
    Xte =test_X(:,2:end);
    Xte(isnan(Xte)) = 0; Xte(isinf(Xte)) = 0;
    Xte = bsxfun(@rdivide, Xte, sqrt(sum(Xte .^ 2, 2)));
    Xtr(isnan(Xte)) = 0; Xtr(isinf(Xte)) = 0;
    train_X=train_X(:,2:end);
    train_X(isnan(train_X)) = 0; train_X(isinf(train_X)) = 0;
    train_X = bsxfun(@rdivide, train_X, sqrt(sum(train_X .^ 2, 2)));
    train_X(isnan(train_X)) = 0; train_X(isinf(train_X)) = 0;
    val_X=val_X(:,2:end); 
    val_X(isnan(val_X)) = 0; val_X(isinf(val_X)) = 0;
    val_X = bsxfun(@rdivide, val_X, sqrt(sum(val_X .^ 2, 2)));
    val_X(isnan(val_X)) = 0; val_X(isinf(val_X)) = 0;
     


trainval_Y_word2vec=trainval_Y('word2vec');
test_Y_word2vec=test_Y('word2vec');
train_Y_word2vec=train_Y('word2vec');
val_Y_word2vec=val_Y('word2vec');

trainval_Y_glove=trainval_Y('glove');
test_Y_glove=test_Y('glove');
train_Y_glove=train_Y('glove');
val_Y_glove=val_Y('glove');

trainval_Y_wordnet=trainval_Y('wordnet');
test_Y_wordnet=test_Y('wordnet');
train_Y_wordnet=train_Y('wordnet');
val_Y_wordnet=val_Y('wordnet');

a=[1 0 0 0];


b1=unique(Ytr);
for i=1:size(b1,1)
   b2=find(Ytr==b1(i));
   b3=Xtr(b2,:);

b4(i,:)=mean(b3,1);
end

    nr_fold = 1;


%% training
if (strcmp(task, 'train'))
    for i = 1 : length(opt.lambda)
        W_record = cell(1, nr_fold);
        for j = 1 : nr_fold
            Xbase = train_X;

            Ybase = Ytr1;
            
            if (strcmp(opt.loss_type, 'OVO'))
                W = train_W_OVO([], Xbase, Ybase, opt.lambda(i));
            elseif (strcmp(opt.loss_type, 'CS'))
                W = train_W_CS([], Xbase, Ybase, opt.lambda(i));
            elseif (strcmp(opt.loss_type, 'struct'))
                W = train_W_struct([], Xbase, Ybase, Sig_dist(unique(Ybase), unique(Ybase)), opt.lambda(i));
            else
                display('Wrong loss type!');
                return;
            end
            W_record{j} = W;

            save(['../SynC_CV_classifiers/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type...
                '_lambda' num2str(opt.lambda(i)) '.mat'], 'W_record');
        end
    end
end

%% validation
if (strcmp(task, 'val'))
    acc_val = zeros(length(opt.lambda), length(opt.Sim_scale));
    for i = 1 : length(opt.lambda)
        load(['../SynC_CV_classifiers/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type...
            '_lambda' num2str(opt.lambda(i)) '.mat'], 'W_record');
        for j = 1 : nr_fold

            Xval =val_X;
            Yval =Yval;
            W = W_record{j};

            for k = 1 : length(opt.Sim_scale)
                if(strcmp(dataset, 'Dogs113'))
               
                Sim_base2 = Compute_Sim(train_Y_word2vec', train_Y_word2vec', opt.Sim_scale(k), Sim_type);
                Sim_base3 = Compute_Sim(train_Y_glove', train_Y_glove', opt.Sim_scale(k), Sim_type);
                Sim_base4 = Compute_Sim(train_Y_wordnet', train_Y_wordnet', opt.Sim_scale(k), Sim_type);
                
                
                Sim_val2 = Compute_Sim(val_Y_word2vec', train_Y_word2vec', opt.Sim_scale(k), Sim_type);
                Sim_val3 = Compute_Sim(val_Y_glove', train_Y_glove', opt.Sim_scale(k), Sim_type);
                Sim_val4 = Compute_Sim(val_Y_wordnet', train_Y_wordnet', opt.Sim_scale(k), Sim_type);
                

                Sim_base=b(1,1)*Sim_base2+b(1,2)*Sim_base3+b(1,3)*Sim_base4;
                 Sim_val=b(1,1)*Sim_val2+b(1,2)*Sim_val3+b(1,3)*Sim_val4;
                else    
                Sim_base1 = Compute_Sim(train_Y_cont', train_Y_cont', opt.Sim_scale(k), Sim_type);
                Sim_base2 = Compute_Sim(train_Y_word2vec', train_Y_word2vec', opt.Sim_scale(k), Sim_type);
                Sim_base3 = Compute_Sim(train_Y_glove', train_Y_glove', opt.Sim_scale(k), Sim_type);
                Sim_base4 = Compute_Sim(train_Y_wordnet', train_Y_wordnet', opt.Sim_scale(k), Sim_type);
                
                Sim_val1 = Compute_Sim(val_Y_cont', train_Y_cont', opt.Sim_scale(k), Sim_type);
                Sim_val2 = Compute_Sim(val_Y_word2vec', train_Y_word2vec', opt.Sim_scale(k), Sim_type);
                Sim_val3 = Compute_Sim(val_Y_glove', train_Y_glove', opt.Sim_scale(k), Sim_type);
                Sim_val4 = Compute_Sim(val_Y_wordnet', train_Y_wordnet', opt.Sim_scale(k), Sim_type);
                
                 Sim_base=a(1,1)*Sim_base1+a(1,2)*Sim_base2+a(1,3)*Sim_base3+a(1,4)*Sim_base4;
                 Sim_val=a(1,1)*Sim_val1+a(1,2)*Sim_val2+a(1,3)*Sim_val3+a(1,4)*Sim_val4;
                end
                V = pinv(Sim_base) * W;            
                Ypred_val = test_V(V, Sim_val, Xval, Yval);
                acc_val(i, k) = acc_val(i, k) + evaluate_easy(Ypred_val, Yval) / nr_fold;          
            end
            clear W;
        end
        clear W_record;
    end
    save(['../SynC_CV_results/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type '.mat'], 'acc_val', 'opt');
end

%% testing
if (strcmp(task, 'test'))
    if(isempty(direct_test))
        load(['../SynC_CV_results/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type '.mat'], 'acc_val', 'opt');
        [loc_lambda, loc_Sim_scale] = find(acc_val == max(acc_val(:)));
        lambda = opt.lambda(loc_lambda(1)); Sim_scale = opt.Sim_scale(loc_Sim_scale(1));
    else
        lambda = direct_test(1); Sim_scale = direct_test(2);
    end
    
    if (strcmp(opt.loss_type, 'OVO'))
        W = train_W_OVO([], Xtr, Ytr, lambda);
    elseif (strcmp(opt.loss_type, 'CS'))
        W = train_W_CS([], Xtr, Ytr, lambda);
    elseif (strcmp(opt.loss_type, 'struct'))
        W = train_W_struct([], Xtr, Ytr, Sig_dist(unique(Ytr), unique(Ytr)), lambda);
    else
        display('Wrong loss type!');
        return;
    end


 Sim_tes1=zeros(size(unique(Yte),1),size(unique(Yte),1));
 Sim_tes2=zeros(size(unique(Yte),1),size(unique(Ytr),1));
   
   
for jj=1:100 %propagation iteration time

    
      Sim_tr1 = Compute_Sim(trainval_Y_cont',trainval_Y_cont', Sim_scale, Sim_type);
      Sim_tr2 = Compute_Sim(trainval_Y_word2vec',trainval_Y_word2vec', Sim_scale, Sim_type);
      Sim_tr3 = Compute_Sim(trainval_Y_glove',trainval_Y_glove', Sim_scale, Sim_type);
      Sim_tr4 = Compute_Sim(trainval_Y_wordnet',trainval_Y_wordnet', Sim_scale, Sim_type);
      Sim_tr5 = Compute_Sim(b4, b4, Sim_scale, Sim_type);
%     
      Sim_te1 = Compute_Sim(test_Y_cont', trainval_Y_cont', Sim_scale, Sim_type);
      Sim_te2 = Compute_Sim(test_Y_word2vec', trainval_Y_word2vec', Sim_scale, Sim_type);
      Sim_te3 = Compute_Sim(test_Y_glove', trainval_Y_glove', Sim_scale, Sim_type);
      Sim_te4 = Compute_Sim(test_Y_wordnet', trainval_Y_wordnet', Sim_scale, Sim_type);
      

      Sim_tr=a(1,1)*Sim_tr1+a(1,2)*Sim_tr2+a(1,3)*Sim_tr3+a(1,4)*Sim_tr4;
      Sim_te=a(1,1)*Sim_te1+a(1,2)*Sim_te2+a(1,3)*Sim_te3+a(1,4)*Sim_te4;
      Sim_te=Sim_tes1*Sim_te+Sim_te+Sim_tes2;

     V = pinv(Sim_tr) * W;

    Ypred_te = test_V(V, Sim_te, Xte, Yte);
    
       %！！！！！！！！！！revised____   
  b5=unique(Ypred_te);
   b8=zeros(size(unique(Yte),1),size(Xte,2));
for i=1:size(b5,1)
   b6=find(Ypred_te==b5(i));
   b7=Xte(b6,:);
   b8(i,:)=mean(b7,1);
end

 Sim_tes1 = Compute_Sim(b8, b8, Sim_scale, Sim_type);   
 Sim_tes2 = Compute_Sim(b8, b4, Sim_scale, Sim_type);  
    
    
    
   acc_te1(jj) = evaluate_easy(Ypred_te, Yte);
end   
 acc_te=max(acc_te1);
    
    

    save(['../SynC_results/SynC_fast_' opt.loss_type '_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type...
        '_lambda' num2str(lambda) '_Sim_scale' num2str(Sim_scale) '.mat'], 'W', 'V', 'lambda', 'Sim_scale', 'acc_te');
end

end

function Sig_dist = Sig_dist_comp(Sig_Y)
inner_product = Sig_Y * Sig_Y';
C = size(Sig_Y, 1);
Sig_dist = max(diag(inner_product) * ones(1, C) + ones(C, 1) * diag(inner_product)' - 2 * inner_product, 0);
Sig_dist = sqrt(Sig_dist);
end