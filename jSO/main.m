clc; clear all;
mex cec17_func.cpp -DWINDOWS

format long;
Runs=51;
fhd=@cec17_func;
NP = 100;

func=[1:30];
optimum=100*func;


% for i=1:29
% eval(['load input_data/shift_data_' num2str(i) '.txt']);
% eval(['O=shift_data_' num2str(i) '(1:10);']);
% f(i)=cec14_func(O',i);i,f(i)
% end


for D= [10 30 50 100]
    switch D
        case 10
            max_nfes=100000;
        case 30
            max_nfes=300000;
        case 50
            max_nfes=500000;
        case 100
            max_nfes=1000000;
        otherwise
            disp('Error..')
    end
    fprintf('\n-------------------------------------------------------\n\n')
    jso(Runs,fhd,D,func,max_nfes,NP,optimum);
end