function result=calculateLogDensity_single(Kernal,xVar, f, theta)
      nTime=Kernal.nTime;                         
      DashCs=Kernal.DashC;
      CPhis=Kernal.CPhi;
      sigmas=Kernal.obsNoise;
      As=Kernal.As;
      Lambda=Kernal.gamma;
      includeDet=Kernal.includeDet;

      x=xVar.x;
      y=xVar.y;
      if Kernal.standardize==1
         mean_single=mean(y);
         std_single=std(y);
         means=y*0.0+mean_single;
         stds=y*0.0+std_single;
      elseif Kernal.normalize==1
         mean_single=mean(y);
         std_single=std(y);
         means=y*0.0+mean_single;
         stds=y*0.0+1.0;
      else
          means=y*0.0;
          stds=y*0.0+1.0;
      end

      
      prob = calculateFPostLog_single(f,x,theta,As,DashCs,CPhis,Lambda,means, stds, includeDet)...
            +calculateGPPostLog_single(nTime,y,x,CPhis,sigmas,means,stds,includeDet);
                         
      scale= 1.0;
      result=prob/scale;                   
end

function result=calculateFPostLog_single(f,x,theta,As,DashCs,CPhis,Lambda,xmean, xstd, includeDet)
    xPret=x;
    xNormal=xstd.*xPret+xmean;
    fMatrix=f;

    currentMean=DashCs*(CPhis\xPret);
    currentDiff=(fMatrix-currentMean)/std(x); 
    currentSigma=As+Lambda*eye(length(x));
    probabilities=-0.5*dot(currentDiff,(currentSigma\currentDiff));
    if includeDet==1
        detTemp=det(currentSigma);
        detArgumentProb=-0.5*log(abs(detTemp))*sign(detTemp);
        probabilities=probabilities+detArgumentProb;
    end
    result=probabilities;
end

function result=calculateGPPostLog_single(nTime,y,x,CPhis,sigmas,ymean,ystd,includeDet)
    yPret = y;
    currentXPret=x; 
    currentYPret=yPret; 
    
    currentXPret2=(x-mean(y))/std(y);
    priorContrib = -0.5*dot(currentXPret2,(CPhis\currentXPret2));
    
    if includeDet==1
        detTemp=det(CPhis);
        detArgumentPrior=-0.5*log(abs(detTemp))*sign(detTemp);
        priorContrib=priorContrib+detArgumentPrior;
    end
    difference = (currentXPret-currentYPret)/std(y);
    obsContrib= -0.5/(sigmas.^2.0)*dot(difference,difference);
    if includeDet==1
        detTemp=det(sigmas*sigmas*eye(nTime));
        detArgumentObs=-0.5*log(abs(detTemp))*sign(detTemp);
        obsContrib=obsContrib+detArgumentObs;
    end
    result=priorContrib+obsContrib;

end


function Ds=getDs(DashCs,CPhis,x)
    Ds=DashCs*(CPhis\x);
end

