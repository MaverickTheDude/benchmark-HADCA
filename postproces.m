% import wynikow z pliku
PATH = '\\wsl$\Ubuntu-20.04\home\maverick\code\examplecp';
working_dir = cd(PATH);
file = 'results.txt';
data = importdata(file);
Nbodieas = (size(data,1)-1)/2;

T = data(1,:);
pTab = data(2:Nbodieas+1,:);
qTab = data(end-Nbodieas+1:end,:);

cd(working_dir);
close all; figure('color','white');
plot(T,pTab,T,qTab); grid on

%% check dla adjointa (joint-space vs global)
PATH = '\\wsl$\Ubuntu-20.04\home\maverick\code\examplecp\output';
working_dir = cd(PATH);
file  = 'resultsAdjoint.txt';
fileG = 'resultsAdjointGlobal.txt';
data  = importdata(file);
dataG = importdata(fileG);
Nb    = (size(data,1)-4)/2; % -4 since time & 3 x norms
n     = (size(dataG,1)-4)/2; % -4 since time & 3 x norms

T = data(1,:);
e = data(2 : Nb+1, :);
c = data(Nb+2 : 2*Nb+2, :);
eta = dataG(2 : n+1, :);
ksi = dataG(n+2 : 2*n+2, :);

cd(working_dir);
close all; figure('color','white');
plot(T,c(1,:), T,e(1,:)); grid on; title('joint space')
legend('c','e')
figure('color','white');
plot(T, data(end-2:end, :)); grid on; title('norms: joint space')

figure('color','white');
plot(T,ksi(1,:), T,eta(1,:)); grid on; title('global formulation')
legend('ksi','eta')
figure('color','white');
plot(T, dataG(end-2:end, :)); grid on; title('norms: global formulation')