clear
clc
load('data_ddos.mat')
rng(1);
ind=1:1075;
randind=randperm(1075);

k=1;
%indinp=[14:17,19:19];
indinp=[1:19];
indtr=randind(1:800);
indts=randind(801:1075);
tout=table(data_label(indtr,2));
tout.Properties.VariableNames(1)={'out'};
tin=array2table(data_ddos(indtr,indinp));

tbl = [tin tout];
c = cvpartition(tbl.out,"Holdout",0.20);

tbltr = tbl(training(c),:);
tblval = tbl(test(c),:);


t = templateSVM('Standardize',true,'KernelFunction','gaussian');
clear all;clc
addpath('codes','dataset');
Load data
D=load('spambase.data');
A=D(:,1:57);             % Inputs
B=D(:,58);               % Targets
define Options
Opts.ELM_Type='Class';    % 'Class' for classification and 'Regrs' for regression
Opts.number_neurons=200;  % Maximam number of neurons
Opts.Tr_ratio=0.70;       % training ratio
Opts.Bn=1;                % 1 to encode  lables into binary representations
                          % if it is necessary
Training
[net]= elm_LB(A,B,Opts);
 net
net = 

           bn: 'binary Targets'
          app: 'Classification'
            X: [3220x57 double]
            Y: [3220x1 double]
          Xts: [1381x57 double]
          Yts: [1381x1 double]
           IW: [200x57 double]
           OW: [200x2 double]
        Y_hat: [3220x1 double]
      Yts_hat: [1381x1 double]
      BnY_hat: [3220x2 double]
    BnYts_hat: [1381x2 double]
          min: 0
          max: 1
         Opts: [1x1 struct]
       tr_acc: 0.8814
       ts_acc: 0.8689

prediction
[output]=elmPredict(net,A);

SVMModel = fitcgam(tbltr,'out');
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
C = confusionmat(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
statsOfMeasure(C, 0)

%%

k=2;
%indinp=[14:17,19:19];
indinp=[1:19];
indtr=randind([1:600, 801:1075]);
indts=randind([601:800]);
tout=table(data_label(indtr,2));
tout.Properties.VariableNames(1)={'out'};
tin=array2table(data_ddos(indtr,indinp));

tbl = [tin tout];
c = cvpartition(tbl.out,"Holdout",0.20);

tbltr = tbl(training(c),:);
tblval = tbl(test(c),:);


%t = templateSVM('Standardize',true,'KernelFunction','gaussian');

SVMModel = fitctree(tbltr,'out');
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
C = confusionmat(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
statsOfMeasure(C, 0)
%%
k=3;
%indinp=[14:17,19:19];
indinp=[1:19];
indtr=randind([1:400, 601:1075]);
indts=randind([401:600]);
tout=table(data_label(indtr,2));
tout.Properties.VariableNames(1)={'out'};
tin=array2table(data_ddos(indtr,indinp));

tbl = [tin tout];
c = cvpartition(tbl.out,"Holdout",0.20);

tbltr = tbl(training(c),:);
tblval = tbl(test(c),:);


%t = templateSVM('Standardize',true,'KernelFunction','gaussian');

SVMModel = fitctree(tbltr,'out');
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
C = confusionmat(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
statsOfMeasure(C, 0)

%%
k=4;
%indinp=[14:17,19:19];
indinp=[1:19];
indtr=randind([1:200, 401:1075]);
indts=randind([201:400]);
tout=table(data_label(indtr,2));
tout.Properties.VariableNames(1)={'out'};
tin=array2table(data_ddos(indtr,indinp));

tbl = [tin tout];
c = cvpartition(tbl.out,"Holdout",0.20);

tbltr = tbl(training(c),:);
tblval = tbl(test(c),:);


%t = templateSVM('Standardize',true,'KernelFunction','gaussian');

SVMModel = fitctree(tbltr,'out');
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
C = confusionmat(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
statsOfMeasure(C, 0)
%%
k=5;
%indinp=[14:17,19:19];
indinp=[1:19];
indtr=randind([201:1075]);
indts=randind([1:200]);
tout=table(data_label(indtr,2));
tout.Properties.VariableNames(1)={'out'};
tin=array2table(data_ddos(indtr,indinp));

tbl = [tin tout];
c = cvpartition(tbl.out,"Holdout",0.20);

tbltr = tbl(training(c),:);
tblval = tbl(test(c),:);


%t = templateSVM('Standardize',true,'KernelFunction','gaussian');

SVMModel = fitctree(tbltr,'out');
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
C = confusionmat(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
statsOfMeasure(C, 0)