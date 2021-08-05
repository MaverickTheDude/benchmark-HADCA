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