function Kernal=KernalSet(theta,sigma,nugget,gamma,time,normalize,standardize,includeDet)
    ker_params.theta=theta;
    ker_params.sigma=sigma;
    ker_params.nugget=nugget;

    Kernal.time=time;
    Kernal.nTime=length(time);

    [C_Phi,DashC_Phi,CDash_Phi,C_PhiDoubleDash]=getCPhi(ker_params,time);

    Kernal.CPhi=C_Phi;
    Kernal.CDash=CDash_Phi;
    Kernal.DashC=DashC_Phi;
    Kernal.CDoubleDash=C_PhiDoubleDash;

    Kernal.obsNoise=sigma; 
    Kernal.As=getAs(CDash_Phi, DashC_Phi, C_Phi, C_PhiDoubleDash);
    Kernal.gamma=gamma;
    Kernal.normalize=normalize;
    Kernal.standardize=standardize;
    Kernal.includeDet=includeDet;

end



function result=ker(ker_params,time1,time2)
    sigmaFSq=ker_params.theta(1);
    result=sigmaFSq*asin(getZ(ker_params,time1,time2));
end

function result=getZ(ker_params,time1,time2)
    a=ker_params.theta(2);
    b=ker_params.theta(3);
    result=(a+b*time1*time2)./getZNorm(ker_params,time1,time2);
end

function result=getZNorm(ker_params,time1,time2)
    a=ker_params.theta(2);
    b=ker_params.theta(3);
    result=sqrt((a+b*time1*time1+1.0)*(a+b*time2*time2+1.0));
end

function result=getDZDt1(ker_params, time1, time2)
    % derivative of Z w.r.t. time1 
    a=ker_params.theta(2);
    b=ker_params.theta(3);
    firstSummand = b*time2./getZNorm(ker_params,time1, time2);
    secondSummand = b*time1*getZ(ker_params,time1, time2)/(a + b*time1*time1 + 1.0);
    result=firstSummand - secondSummand;
end


function result=getDZDt2(ker_params, time1, time2)
    %   derivative of Z w.r.t. time2 
    a=ker_params.theta(2);
    b=ker_params.theta(3);
    firstSummand = b*time1/getZNorm(ker_params,time1, time2);
    secondSummand = b*time2*getZ(ker_params,time1, time2)./(a + b*time2*time2 + 1.0);
    result=firstSummand - secondSummand;
end

function result=getDZDt1Dt2(ker_params, time1, time2)
    % second order derivative of Z w.r.t. time1 and time2 
    a=ker_params.theta(2);
    b=ker_params.theta(3);
    firstSummand = b ./ getZNorm(ker_params,time1, time2);
    secondSummand = -b*b*time2*time2./(getZNorm(ker_params,time1,time2)*(a + b*time2*time2 + 1.0));
    thirdSummand = -b*time1/(a + b*time1*time1 + 1.0) .* getDZDt2(ker_params,time1,time2);
    result=firstSummand + secondSummand + thirdSummand;
end

function result=CDash(ker_params, time1, time2)
 
    %         returns the derivative of the correlation between time1 and time2 with
    %         respect to time2, used in the C_Phi' matrix
    %         
    sigmaFSq = ker_params.theta(1);
    result=sigmaFSq/sqrt(1-getZ(ker_params,time1, time2).^2.0)*getDZDt2(ker_params,time1, time2);
end


function result=DashC(ker_params, time1, time2)

    sigmaFSq = ker_params.theta(1);
    result=sigmaFSq /sqrt(1-getZ(ker_params,time1, time2).^2.0)*getDZDt1(ker_params,time1, time2);
end

function result=CDoubleDash(ker_params, time1, time2)

    sigmaFSq = ker_params.theta(1);
    result=sigmaFSq / sqrt(1-getZ(ker_params,time1, time2).^2.0) * ...
               (getZ(ker_params,time1, time2)./(1-getZ(ker_params,time1, time2).^2.0) * ...
                getDZDt1(ker_params,time1, time2)*getDZDt2(ker_params,time1, time2) ...
                +getDZDt1Dt2(ker_params,time1, time2) );
end


function [C_Phi,DashC_Phi,CDash_Phi,C_PhiDoubleDash]=getCPhi(ker_params,time)
    NN=length(time);
    C_Phi=zeros(NN,NN);
    DashC_Phi=zeros(NN,NN);
    CDash_Phi=zeros(NN,NN);
    C_PhiDoubleDash=zeros(NN,NN);
    for j = 1:NN
        for i=1:NN
            C_Phi(i,j)=ker(ker_params,time(i),time(j));
            DashC_Phi(i,j)=DashC(ker_params,time(i),time(j));
            CDash_Phi(i,j)=CDash(ker_params,time(i),time(j));
            C_PhiDoubleDash(i,j)=CDoubleDash(ker_params,time(i),time(j));
        end
    end
    C_Phi=C_Phi+ker_params.nugget*eye(NN);
end


function As=getAs(CDashs, DashCs, CPhis, CDoubleDashs)
    As=CDoubleDashs-DashCs*(CPhis\CDashs);    
end






