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

SVMModel = fitcecoc(tbltr,'out','Learners',t,'FitPosterior',true);
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


t = templateSVM('Standardize',true,'KernelFunction','gaussian');

SVMModel = fitcecoc(tbltr,'out','Learners',t,'FitPosterior',true);
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)))
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



t = templateSVM('Standardize',true,'KernelFunction','gaussian');

SVMModel = fitcecoc(tbltr,'out','Learners',t,'FitPosterior',true);
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)))
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


t = templateSVM('Standardize',true,'KernelFunction','gaussian');

SVMModel = fitcecoc(tbltr,'out','Learners',t,'FitPosterior',true);
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)))
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



t = templateSVM('Standardize',true,'KernelFunction','gaussian');

SVMModel = fitcecoc(tbltr,'out','Learners',t,'FitPosterior',true);
%CVSVMModel = crossval(SVMModel)
confusionchart(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)))
C = confusionmat(data_label(indts,2),predict(SVMModel,data_ddos(indts,indinp)));
statsOfMeasure(C, 0)

