clc; clear all; close all;


%% Uploading Data Sets



Data    = xlsread('Orginal Data.xlsx','Sheet1');
Choice  = 0.7;

Np      = 5; % Total Number of Inputs

% Note: Be careful about your input data.

%% Data Randomization


xx = 0;
nn = 0;
% while  xx < 20
%     for yy = 70:1:90
    
% xx = xx + 10;

% for xx = 100000:1:1000000
rand('seed',487)       % To get same results every time, Change numeric value to get different results 


Length_Data          =      length(Data);
Random_Nums          =      randperm(Length_Data,Length_Data);
Random_Nums          =      Random_Nums'; 



Data_random         =       Data  (Random_Nums, :);
Data_Tr             =       Data_random(1 : ceil(length(Data) * Choice), :);
Data_Tst            =       Data_random(length(Data_Tr)+1 : end, :);


Data_Tr             =       sortrows(Data_Tr,1);
Data_Tst            =       sortrows(Data_Tst,1);



Training_Depth      =       Data_Tr(:,1); 
Testing_Depth       =       Data_Tst(:,1);

Input_Training      =       Data_Tr(:, 2 :Np);     % Don't take depths as an input .
Output_Training     =       Data_Tr(:,Np + 1);
Input_test          =       Data_Tst(:, 2 :Np) ;
Output_test         =       Data_Tst(:,Np + 1) ;

% To be sure, check your inputs and outputs at this point.


% Creating a Fuzzy Logic Network
%% Genfis 1 -- Grid Partionining Main Parameters

nMFs            = 5;            % Number of Membership Function
InputMF         = 'gaussmf';    % Type of Input Membership Function
OutputMF        = 'linear' ;    % Type of Output Membership Function


%% Genfis 2 -- Subtractive Clustring Main Parameters

%Radius          =       0.8; %% Cluster Radius
Ep_Size         =       100;  %% Number of Itterations


%% Genfis 1 Command

fis                   =     genfis1([Input_Training Output_Training],nMFs,InputMF,OutputMF);

%% Genfis 2 Command

%fis                     =     genfis2(Input_Training,Output_Training,Radius);


%% Common commands use in both cases


fis                     =       anfis([Input_Training,Output_Training],fis,Ep_Size);

Results_Training        =       evalfis(Input_Training,fis);
Results_Testing         =       evalfis(Input_test,fis);


CorrCoef_Train          =       corrcoef(Results_Training,Output_Training); 
CorrCoef_Test           =       corrcoef(Results_Testing,Output_test);

% AAPE_Train1             =       ( abs ( ( Output_Training - Results_Training)./Output_Training))*100;
% AAPE_Test1              =       ( abs ( ( Output_test - Results_Testing)./Output_test))*100;

nn = nn + 1

AAPE_Train(nn)          =       mean ( abs ( ( Output_Training - Results_Training)./Output_Training))*100;
AAPE_Test(nn)           =       mean ( abs ( ( Output_test - Results_Testing)./Output_test))*100;
AAPE_Test1 = AAPE_Test(nn)
% xlswrite ('AAPE_Test.xlsx',AAPE_Test)
%     end   
% end

figure (1)
plot(Results_Training,Training_Depth,'--')
hold on;
plot(Output_Training,Training_Depth,'r*');
legend('Predicted','Real');
set (gca,'Ydir','reverse')
xlabel('SFT(C)','FontSize',12,'FontWeight','bold','Color','k');
ylabel('Depth m','FontSize',12,'FontWeight','bold','Color','k');
grid on
title('SFT vs Depth for Training Data using ANFIS (Subtractive Clustring)','FontSize',12,'FontWeight','bold','Color','k');


figure (2)
plot(Results_Testing,Testing_Depth,'--')
hold on;
plot(Output_test,Testing_Depth,'ro');
legend('Predicted','Real');
set (gca,'Ydir','reverse')
xlabel('SFT(C)','FontSize',12,'FontWeight','bold','Color','k');
ylabel('Depth m','FontSize',12,'FontWeight','bold','Color','k');
grid on
title('SFT vs Depth for Testing Data using ANFIS (Subtractive Clustring)','FontSize',12,'FontWeight','bold','Color','k');

figure (3)
plotregression(Output_Training,Results_Training)
xlabel('Actual SFT(C)','FontSize',12,'FontWeight','bold','Color','k');
ylabel('Predicted SFT(C)','FontSize',12,'FontWeight','bold','Color','k');
title('SFT Prediction Training Data using ANFIS (Subtractive Clustring)');


figure (4)
plotregression(Output_test,Results_Testing)
xlabel('Actual SFT (C)','FontSize',12,'FontWeight','bold','Color','k');
ylabel('Predicted SFT (C)','FontSize',12,'FontWeight','bold','Color','k');
title('SFT Prediction Testing Data using ANFIS (Subtractive Clustring)');

CorrCoef_Train
CorrCoef_Test
AAPE_Train
AAPE_Test


