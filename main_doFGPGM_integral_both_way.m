clc;
clear;
close all;


trueTheta=[0.5;1.0];
XInit = [0.5;0.0;exp(0.5)];



time = (0:0.2:5)';

[tt,obs0] = ode45(@(tt,yb) odefcn(tt,yb,trueTheta), time, XInit);
[n1,n2]=size(obs0);
rng(0,'twister'); 
obs = obs0 + 0.2*randn(n1,n2);


flag_un_num=2;

EqNum=length(obs(1,:));

gammaValue=1.e-3;
gammas=gammaValue*ones(EqNum,1);


numParams=length(trueTheta);
theta0=abs(randn(numParams,1));

thetaMagnitudes=zeros(numParams,1);

stateStds=0.001;
paramStds=0.1;


GPPosteriorInit=1;
normalize=0;includeDet=0;standardize = 0;

% ========== Set the Latent Variable===============
coef_known(1:3)=1;
coef_known(1:2)=0;

%========================  GP regression ==================================
    for k=1:EqNum
       if coef_known(k)>-0.5
            y_obs=obs(:,k);
            opts = statset('fitrgp');
            opts.TolFun = 1e-12; 
            opts.TolX =1.e-12;
           
            enlargeCoef =1.0;
            gprMdl3 = fitrgp(time,enlargeCoef*y_obs,'KernelFunction','matern52','Optimizer','quasinewton','Basis','none', ...  
                      'FitMethod','exact','PredictMethod','exact');
                  
            ypred3 = resubPredict(gprMdl3);
               

            sigmaM = gprMdl3.KernelInformation.KernelParameters;
            gprsigma = gprMdl3.Sigma;

            csvwrite(['hyperparams/theta2',num2str(k-1),'.csv'],sigmaM);
            csvwrite(['hyperparams/sigma2',num2str(k-1),'.csv'],gprsigma);

            figure(11+k)
            plot(time,y_obs,'r.','linewidth',1.25);hold on;
            plot(time,ypred3/enlargeCoef,'b-','linewidth',1.25);hold on;
            legend('Data','matern52');

            clear gprMdl3 ypred3;
       end
    end

    
%==================FGPGM -Step=============================================

for i=1:EqNum
    %if coef_known(i)==1
        kernal_theta=load(['hyperparams/theta2',num2str(i-1),'.csv']);
        kernal_sigma=load(['hyperparams/sigma2',num2str(i-1),'.csv']);
        kernal_nugget=1.e-4;
        Kernal(i)=KernalSet(kernal_theta,kernal_sigma,kernal_nugget,gammas(i),time,normalize,standardize,includeDet);
    %end
        vectortemp=obs(:,i);
        xVar(i).y=obs(:,i);
        xVar(i).x=obs(:,i);
        xVar(i).lowerBounds=0.0*vectortemp-5.0;
        xVar(i).upperBounds=0.0*vectortemp+5.0;
        xVar(i).proposalStds=0.0*vectortemp+stateStds;
        clear vectortemp;
end
        vectortemp=obs(:,3);
        xVar(3).upperBounds=0.0*vectortemp+15.0;

for i=1:numParams
    ParamObj(i).paramval=theta0(i,1);
    ParamObj(i).lowerBound=0.01;
    ParamObj(i).upperBound=4.0;
    ParamObj(i).proposalStd=paramStds;
end

nSamples=3000;
nBurnin=2000;


tic
[theta,xVar_final,nAccepted,nRejected]=MCMCSampler(XInit,EqNum,coef_known,numParams,Kernal,ParamObj,xVar,nSamples,nBurnin);
toc
   [tt,yb] = ode45(@(tt,yb) odefcn(tt,yb,theta), time, XInit);
   [ta,yb2] = ode45(@(ta,yb2) odefcn(ta,yb2,trueTheta), time, XInit);
   
% for i=1:EqNum
%    figure()
%    plot(Kernal(i).time,xVar_final(i).x,'bo',Kernal(i).time,xVar_final(i).y,'rx',tt,yb(:,i),'k.',tt2,yb2(:,i),'r-'); 
% end

%======================Plot================================================
for i=1:EqNum
    figure()
    set(gcf,'position',[400,200,400,300]);
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [4 3]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 4 3]);
      
   plot(ta,yb2(:,i),'r-','linewidth',1);hold on;
   if(coef_known(i)>0.5)
   plot(ta,obs(:,i),'x','Color',[0.0 0.5 0.0],'linewidth',1.5,'markersize',6);hold on;
   end

   plot(Kernal(i).time,xVar_final(i).x,'b.','linewidth',1.5,'markersize',10);hold on;

   if (coef_known(i)<0.5)
    legend({'true','FGPGM'},'Location','Best');ylabel(['x_',num2str(i)]);xlabel('t');
   else
    legend({'true','observation','FGPGM'},'Location','Best');ylabel(['x_',num2str(i)]);xlabel('t');
   end
   saveas(gcf,['Eg_state_newcode_y',num2str(i),'.png']); 
  %close all;
  
  optstates(:,i)=xVar_final(i).x;
end

paramindex=1:numParams;

    figure()
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [4 3]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 4 3]);

   plot(paramindex(1:numParams),trueTheta(1:numParams),'rd',paramindex(1:numParams),theta(1:numParams),'ko','linewidth',1.5);hold on;
   xlim([paramindex(1)-0.5, paramindex(numParams-0)+0.5]);
   legend({'True Params','FGPGM'},'Location','Best');ylabel('Parameter Index');xlabel('Parameter Value');
  saveas(gcf,['Eg_state_newcode_params.png']); 
  

  
  
%================== Least Square Optimization Step (Optional)==============
LSopt=1;
if LSopt
    options_opt = optimset('MaxIter', 500,'MaxFunEvals',500);
    param_modi=theta;
    [theta_total, f_current0]=fminunc(@(param_modi)(cost_func_cal_num(XInit,obs,time,param_modi,coef_known)),param_modi,options_opt)
    %theta_total=optimal_num(xinit,obs,tt0,params,alpha,step_uni,coef_matrix,coef_matrix_Eq);
    [ta,yb3] = ode45(@(ta,yb3) odefcn(ta,yb3,theta_total), time, XInit);
    
    for i=1:EqNum
        %i=3;
        figure()
        set(gcf, 'PaperUnits', 'inches');
        set(gcf, 'PaperSize', [4 3]);
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperPosition', [0 0 4 3]);
      
        plot(ta,yb2(:,i),'r-','linewidth',1);hold on;
        if(coef_known(i)>0.5)
            plot(ta,obs(:,i),'x','Color',[0.0 0.5 0.0],'linewidth',1.5,'markersize',6);hold on;
        end   
        plot(ta,yb3(:,i),'b-.','linewidth',1);hold on;
        plot(Kernal(i).time,xVar_final(i).x,'b.','linewidth',1.5,'markersize',10);hold on;
        if(coef_known(i)>0.5)
            legend({'true','observation','FGPGM+Opt.','FGPGM'},'Location','Best');xlabel('t');ylabel(['x_',num2str(i)]);
        else
            legend({'true','FGPGM+Opt.','FGPGM'},'Location','Best');ylabel(['x_',num2str(i)]);xlabel('t');
        end
        saveas(gcf,['Eg_state_method1_y',num2str(i),'.png']); 
        close all;
    end
    
    paramindex=1:numParams;
 
    figure()
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [4 3]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 4 3]);

    plot(paramindex(1:numParams),trueTheta(1:numParams),'rd',paramindex(1:numParams),theta_total(1:numParams),'bx',paramindex(1:numParams),theta(1:numParams),'ko','linewidth',1.5);hold on;
    xlim([paramindex(1)-0.5, paramindex(numParams)+0.5]);
    legend({'True Params','FGPGM+Opt.','FGPGM'},'Location','Best');ylabel('Parameter Index');xlabel('Parameter Value');
    saveas(gcf,['Eg_state_method1_params.png']);     
end
  
  
  
  
  
%=========================Main Subroutines=================================

function [theta,xVar_final,nAccepted,nRejected]=MCMCSampler(XInit,EqNum,coef_known,numParams,Kernal,ParamObj,xVar,nSamples,nBurnin)
    xVar_final=xVar;

    %------------------set unfolded array--------------------
    tspan=Kernal(1).time;
    k=0;Xunfold_u=[];
    for i=1:numParams
        k=k+1;
        Xunfold(k)=ParamObj(i).paramval;
        proposalStds(k)=ParamObj(i).proposalStd;
        lowerBounds(k)=ParamObj(i).lowerBound;
        upperBounds(k)=ParamObj(i).upperBound;
    end
    for i=1:EqNum
       if coef_known(i)>0.5
        for j=1:Kernal(i).nTime
            k=k+1;
            Xunfold(k)=xVar(i).x(j);
            proposalStds(k)=xVar(i).proposalStds(j);
            lowerBounds(k)=xVar(i).lowerBounds(j);
            upperBounds(k)=xVar(i).upperBounds(j);
        end
       else
        Xunfold_u=[Xunfold_u,tspan'*0.0+XInit(i)];   
       end 
    end
    xNew=Xunfold;
    xNew_u=Xunfold_u;
    
    logPNew= calculateLogDensity(EqNum,coef_known,numParams,Kernal,xVar,xNew,xNew_u);
    allSamples=0.0*Xunfold;
    allSamples_u=0.0*Xunfold_u;
    
    nAccepted=0.0*Xunfold;
    nRejected=0.0*Xunfold;
    nAcceptedtotal=0;
    
    
    
    for j=1:nSamples+nBurnin
        disp(j);
       
        currSamples=0.0*Xunfold;
        currSamples_u=0.0*Xunfold_u;
        for k=1:length(Xunfold)
            xOld=xNew;
            logPOld=logPNew;
            proposal=getProposal(xOld(k),proposalStds(k),lowerBounds(k),upperBounds(k));
            xPotential=xOld;
            xPotential(k)=proposal;
            
            %xPotential_u=[];
            %tic
            if k<(numParams+1)
                %xNewtemp=xPotential;
                theta=xPotential(1:numParams);
                %opts = odeset('RelTol',1e-2,'AbsTol',1e-3);
                %[ta,yb] = ode45(@(ta,yb) odefcn(ta,yb,theta), tspan, XInit, opts);
                %[t,y] = ode45(@(t,y) myode(t,y,ft,f,gt,g), tspan, ic, opts);
                %tic
                
                flag_int=1;
                if flag_int
                    [ta,yb] = ode45(@(ta,yb) odefcn(ta,yb,theta), tspan, XInit);
                else
                    yb = unobserve_cal(theta,XInit,tspan,xVar);
                end
               % toc
                Xunfold_u_temp=[];
                for i2=1:EqNum
                    if coef_known(i2)<0.5
                        add_temp=yb(:,i2);
                        Xunfold_u_temp=[Xunfold_u_temp,add_temp']; 
                    end                    
                end
                xPotential_u=Xunfold_u_temp;
                clear Xunfold_u_temp add_temp ta yb
            end

            normalizationOldGivenNew = getValidProbability(proposal,...
                            proposalStds(k),lowerBounds(k),upperBounds(k));
            normalizationNewGivenOld = getValidProbability(xOld(k),...
                            proposalStds(k),lowerBounds(k),upperBounds(k));  
            
            logPPotential =calculateLogDensity(EqNum,coef_known,numParams,Kernal,xVar,xPotential,xPotential_u); 
            %toc
            if logPPotential-logPOld > log(normalizationOldGivenNew/normalizationNewGivenOld)
                acceptanceProb=1.0;
            else
                acceptanceProb=exp(logPPotential-logPOld);
                acceptanceProb=acceptanceProb*normalizationNewGivenOld/normalizationOldGivenNew;                
            end
            
            if rand(1)<=acceptanceProb
                xNew = xPotential;
                xNew_u=xPotential_u;
                logPNew=logPPotential;
                nAccepted(k)=nAccepted(k)+1;
                nAcceptedtotal=nAcceptedtotal+1;
            else
                logPNew=logPOld;
                nRejected(k)=nRejected(k)+1;
                
            end
            currSamples=currSamples+xNew;
            currSamples_u=currSamples_u+xNew_u;
            
        end
        currSamples=currSamples/(1.0*length(xNew)-0.0*numParams);

        currSamples_u=currSamples_u/(1.0*length(xNew)-0.0*numParams);

        if j>nBurnin
            allSamples=allSamples+currSamples;       
            allSamples_u=allSamples_u+currSamples_u;
        end        
    end
    allSamples=allSamples/nSamples;
    allSamples_u=allSamples_u/nSamples;
    
    
    k=0;k2=0;
    for i=1:numParams
        k=k+1;
        theta(i)=allSamples(k);
    end
    for i=1:EqNum
       if coef_known(i)>0.5
        for j=1:Kernal(i).nTime
            k=k+1;
            xVar_final(i).x(j)=allSamples(k);
        end
       else
        for j=1:Kernal(i).nTime
            k2=k2+1;
           xVar_final(i).x(j)=allSamples_u(k2);
        end
       end
    end    
    
end

function proposal=getProposal(meanvalue,proposalStd,lowerBound,upperBound)
    maxTry=100;
    for i=1:maxTry
        newX=meanvalue+proposalStd*randn(1);
        if newX<upperBound && newX>lowerBound
            proposal=newX;
            return;
        end
    end
    error('Cannot find proper proposals. Try to change bounds');
end

function proposal=getValidProbability(meanvalue,proposalStd,lowerBound,upperBound)
    proposal=cdf('Normal',upperBound,meanvalue,proposalStd)-cdf('Normal',lowerBound,meanvalue,proposalStd);
end


function result=calculateLogDensity(EqNum,coef_known,numParams,Kernal0,xVar0,xNew,xNew_u)

    Kernal=Kernal0;xVar=xVar0;
    k=0;k2=0;
    for i=1:numParams
        k=k+1;
        theta(i)=xNew(k);
    end
    for i=1:EqNum
       if coef_known(i)>0.5
        for j=1:Kernal(i).nTime
            k=k+1;
            xVar(i).x(j)=xNew(k);
        end
       else
        for j=1:Kernal(i).nTime
            k2=k2+1;
           xVar(i).x(j)=xNew_u(k2);
        end
       end
    end    

    prob=0.0;
    for i=1:EqNum
       if(coef_known(i)>0.5)
        tt=Kernal(i).time;
        for i2=1:EqNum
            %xx(:,i2)=interp1(Kernal(i2).time,xVar(i2).x,tt,'linear');
            xx(:,i2)=xVar(i2).x;
        end

        for j=1:length(tt)
            y_temp(1:EqNum)=xx(j,1:EqNum);
            dydt=odefcn(tt(j), y_temp, theta);
            f(j,1)=dydt(i);
        end
        
        prob=prob+calculateLogDensity_single(Kernal(i),xVar(i), f, theta);
        clear f tt xx
       end
    end

    result=prob/(2.0*(length(xNew)-numParams));
    clear Kernal xVar
end




%==============================================
function dydt = odefcn(t,y,theta)
    dydt = zeros(3,1);
    alpha=theta(1);
    beta=theta(2);

    dydt(1) = alpha*y(2).*exp(-beta*y(1));
    dydt(2) = y(1);
    dydt(3) = alpha*y(2).*exp((1.0-beta)*y(1));
end


%==============================================
function dydt = odefcnFNH(t,y,theta)
    dydt = zeros(2,1);
    alpha=theta(1);
    beta=theta(2);
    phi=theta(3);
    dydt(1) =  phi*( y(1)-y(1)*y(1)*y(1)/3.0+y(2) );
    dydt(2) = -1.0/phi* (y(1)-alpha+beta*y(2) );
end




function yb = unobserve_cal(theta,XInit,tspan,xVar)
    alpha=theta(1);
    beta=theta(2);
    phi=theta(3);

    y(:,1)=xVar(1).x;
    y(:,2)=xVar(2).x;

    yb=0.0*y;

    yb(1,:)=XInit(:);

    for i=2:length(tspan)
        dt=tspan(i)-tspan(i-1);
        %yb(i,1)= yb(i-1,1)+dt*( phi*( yb(i-1,1)-yb(i-1,1).^3/3.0+y(i-1,2) ));
        yb(i,2)= yb(i-1,2)+dt*( -1.0/phi* (y(i-1,1)-alpha+beta*yb(i-1,2) ));
    end
end

function dydt2 = odefcn_eg1(t,y,theta)
    dydt2 = zeros(2,1);
    theta1=theta(1);
    theta2=theta(2);
    theta3=theta(3);
    theta4=theta(4);
    dydt2(1) =  theta1*y(1)-theta2*y(1)*y(2);
    dydt2(2) = -theta3*y(2)+theta4*y(1)*y(2);
end

function dydt3 = odefcn_eg2(t, y, theta)
    dydt3 =zeros(5,1);
%         
%         x: (S, S_d, R, RS, Rpp)
%         theta: (k1, k2, k3, k4, V, Km)
%         

        k1 = theta(1);
        k2 = theta(2);
        k3 = theta(3);
        k4 = theta(4);
        V = theta(5);
        Km = theta(6);

        S = y(1);
        R = y(3);
        RS = y(4);
        Rpp = y(5);

        dydt3(1) = -k1*S - k2*S*R + k3*RS;  %dS
        dydt3(2) = k1*S;    %dS_d 
        dydt3(3) = -k2*S*R + k3*RS + V*Rpp/(Km + Rpp + 1e-6);   %dR 
        dydt3(4) = k2*S*R - k3*RS - k4*RS;  %dRS
        dydt3(5) = k4*RS - V*Rpp/(Km + Rpp + 1e-6); %dRpp
end



function ff2=cost_func_cal_num(xinit,obs,tt0,params,coef_matrix_Eq)
    [numPoint,numEq]=size(obs);
    [numParam,temp]=size(params);

     tspan=tt0;

    [ta,ya] = ode45(@(ta,ya) odefcn(ta,ya,params), tspan, xinit);    

    ff2=0.0;
    for i=1:numEq
        diffya=ya(:,i)-obs(:,i);
        if(coef_matrix_Eq(i)>0.5)
            ff2=ff2+L2_compute(tt0,diffya)/L2_compute(tt0,obs(:,i));
        end
    end
end

function L2_value=L2_compute(tt0,yy0)
    [numt,temp]=size(tt0);
    norm_yy0=0.0;
    for i=1:numt-1
        dt=tt0(i+1)-tt0(i);
        norm_yy0=norm_yy0+dt*0.5*(yy0(i+1)*yy0(i+1)+yy0(i)*yy0(i));   
    end
    L2_value=norm_yy0;
end


function pff2pt_total=grad_theta_num(xinit,obs,tt0,params,alpha,step_uni,coef_matrix,coef_matrix_Eq)
    ppstep=0.001;
    [numPoint,numEq]=size(obs);
    [numParam,temp]=size(params);

    for i=1:numParam
        trueparamR=params;
        trueparamR(i)=params(i)+ppstep;
        
        trueparamL=params;
        trueparamL(i)=params(i)-ppstep;

        pff2pt_total(i)=cost_func_cal_num(xinit,obs,tt0,trueparamR,alpha,step_uni,coef_matrix,coef_matrix_Eq)...
                       -cost_func_cal_num(xinit,obs,tt0,trueparamL,alpha,step_uni,coef_matrix,coef_matrix_Eq);
    end
   
end

function theta_total=optimal_num(xinit,obs,tt0,params,alpha,step_uni,coef_matrix,coef_matrix_Eq)
    [numPoint,numEq]=size(obs);
    [numParam,temp]=size(params);
    norm=1.0;
    f_change=10.0;
    
    f_current= cost_func_cal_num(xinit,obs,tt0,params,alpha,step_uni,coef_matrix,coef_matrix_Eq);
    theta_total=params;
    
    while f_current>0.01 
        pff2pt_total=grad_theta_num(xinit,obs,tt0,theta_total,alpha,step_uni,coef_matrix,coef_matrix_Eq);
        
        norm=sqrt(sum(pff2pt_total.^2));
        if norm>step_uni
            step=step_uni/norm;
        else
            step=step_uni/norm;
        end 
        
        for i=1:numParam
            theta_total(i)=theta_total(i)-step*pff2pt_total(i)*coef_matrix(i);
        end
       
        ff2=cost_func_cal_num(xinit,obs,tt0,theta_total,alpha,step_uni,coef_matrix,coef_matrix_Eq);

        f_change=f_current-ff2;
        f_current=ff2
    end
end


