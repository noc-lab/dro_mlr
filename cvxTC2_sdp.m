clear all;clc

cvx_solver SDPT3;
rand('seed',12);



M=load('currentreprDiff_mnist.mat');


M=M.data;
[r,c]=size(M)
M=M+(rand(r,c)-0.5)/100000;
M=M'*M/100000;

n=c;


N=load('currentrepr_mnist.mat');


N=N.data;
[r,c]=size(N)
N=N+(rand(r,c)-0.5)/100000;
N=N'*N/100000;



P=load('currentrepr_mnist_clean.mat');


P=P.data;
[r,c]=size(P)
P=P+(rand(r,c)-0.5)/100000;
P=P'*P/100000;


cvx_begin sdp
    variable W(n,n) semidefinite
    %variable Y(n,n) semidefinite
    %minimize( trace(M'*W) )
    minimize( trace(M'*W) )
    %minimize( -trace(W) )
    subject to
        trace(N'*W) >= 1
        trace(P'*W) >= 1
        %Y==W-0.0000001*eye(n)
%             x1'*W*x1 <= 1
%             x2'*W*x2 <= 1
cvx_end

W=full(W);
rank(W)
r
rank(M)



save(['current_W_.mat'],'W')



