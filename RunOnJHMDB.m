splits = 3;

InitPCNN();

datadir = '/media/data/amorsy/Full_JHMDB';

classes = dir(sprintf('%s/splits',datadir));
classes = {classes.name};
classes=classes(~ismember(classes,{'.','..'}));

startTime = tic;
t = datestr(now);
logFile = fopen(sprintf('logs/%s.txt','JHMDB'),'w');
fprintf('Started PCNN for JHMDB on %s\n',t);
fprintf(logFile,'Started PCNN for JHMDB on %s\n',t);
for ci = 1:length(classes)
   className = classes{ci};
   fprintf('%s\n',className); 
   fileID = fopen(sprintf('logs/%s.txt',className),'w');  
   fprintf('Compute PCNN for %d out of %d classes (%s)\n',ci,length(classes),className);
   fprintf(fileID,'Compute PCNN for %d out of %d classes (%s)\n',ci,length(classes),className);
   totalElapsedTime = 0;
   for splitNum=1:splits
	tStart = tic;
	fprintf('Start Computing PCNN for %d split of %d classes (%s)\n',splitNum,ci,className); 
	fprintf(fileID,'Start Computing PCNN for %d split of %d classes (%s) at %s\n',splitNum,ci,className,datestr(now));   
	ComputePCNN(datadir,className,splitNum);
	tElapsed = toc(tStart)/60;
        totalElapsedTime = totalElapsedTime + tElapsed;
	fprintf('Done PCNN for split %d in class %s in %.1f mins\n',splitNum,className,tElapsed);
	fprintf(fileID,'Done PCNN for split %d in class %s in %.1f mins\n',splitNum,className,tElapsed);
   end
   fprintf('Finished PCNN for class %d or %s in %.1f mins\n',ci,className,totalElapsedTime);
   fprintf(fileID,'Finished PCNN for class %d or %s in %.1f mins\n',ci,className,totalElapsedTime);
   fprintf(logFile,'Finished PCNN for class %d or %s in %.1f mins\n',ci,className,totalElapsedTime);
   fclose(fileID);	
end

endTime = toc(startTime);
t = datestr(now);
fprintf('Finished PCNN on JHMDB at %.1f mins on %s\n',endTime/60,t);
fprintf(logFile,'Finished PCNN on JHMDB at %.1f mins on %s\n',endTime/60,t);
fclose(logFile);
