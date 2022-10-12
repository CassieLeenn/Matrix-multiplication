function [time] = matrix_multiplication()
clc;
clear;
A=load('./A.mat');
B=load('./B.mat');
all_time=0;
for i =1:5
    
    tic;
    C=A.A*B.B;
    time=toc
    all_time=all_time+time;
end 

all_time/5

%disp(['time: ',num2str(toc)]);
%disp(['time: ',num2str(tic)]);

%time=num2str(toc);



end

