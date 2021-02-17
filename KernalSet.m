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

    a=ker_params.theta(2);
    b=ker_params.theta(1);
    r = abs(time1 - time2);
    result=a.^2*(1 + sqrt(5)/b*r + 5.0/3.0/b.^2*r.^2)* exp(-sqrt(5)/b*r);
    % matern 32:
    % result=a.^2*(1 + sqrt(3)/b.*r ).* exp(-sqrt(3)/b.*r);
end



function result=CDash(ker_params, time1, time2)

    result=DashC(ker_params,time2,time1);
end


function result=DashC(ker_params, time1, time2)

    a=ker_params.theta(2);
    b=ker_params.theta(1);
    r = abs(time1 - time2);
    result= - a.^2*5./3./b.^2*(time1 - time2)*(1 + sqrt(5)/b*r)*exp(-sqrt(5)/b*r);
    % matern 32:     
    % result= - a.^2*3/b.^2.*(time1 - time2).*exp(-sqrt(3)/b.*r);
end

function result=CDoubleDash(ker_params, time1, time2)

    a=ker_params.theta(2);
    b=ker_params.theta(1);
    r = abs(time1 - time2);
    result= a.^2*5./3./b.^2  * (1 + sqrt(5)/b*r - 5./b.^2*r.^2)* exp(-sqrt(5)/b*r);
    % matern 32:
    % result= a.^2*3/b.^2*( 1 -sqrt(3)/b*r ).*exp(-sqrt(3)/b.*r);         
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






