function [range,u1,u2,xt,yuce,epsilon,delta,rho]=GM(x0)
n=length(x0);
x0 = x0(:);  % 强制为列向量，避免维度错误
lamda=x0(1:n-1)./x0(2:n);
range=minmax(lamda');%级比范围
x1=cumsum(x0);
B=[-0.5*(x1(1:n-1)+x1(2:n)),ones(n-1,1)];
Y=x0(2:n);
u=B\Y;%拟合u（1）=a，u（2）=b
syms x(t)
x=dsolve(diff(x)+u(1)*x==u(2),x(0)==x0(1));%求符号解
xt=vpa(x,6);
u1=u(1);
u2=u(2);
yuce1=subs(x,t,0:n-1);  %已知数据预测值
yuce1=double(yuce1);
yuce=[x0(1),diff(yuce1)];
epsilon=x0'-yuce;%残差，相对误差，级比偏差值
delta=abs(epsilon./x0');
rho=1-(1-0.5*u(1)/(1+0.5*u(1)))*lamda';
end