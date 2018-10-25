function ez_linprog_data_generation(n, m, numPoints, showBar, dataPath)
%Continuously adds more data points to a file.  Can be called iteratively

%Checks for setup file (if no setup file, then construct one)
%Setup files have the following format: first line number of points in training
%second line is number of points in testing
%setupFileName = 'data' + num2str(n) + '_' + num2str(m) + '.setup';
%setupFileID = fopen(setupFileName, 'r+');

%if setupFileID == -1
%    fprintf('No setup file for n = ' + num2str(n) + ' and m = ' + num2str(m) + '. Creating one now...\n');
%    setupFileID = fopen(setupFileName, 'w+');
%    fprintf(setupFileName, '%f\n%f', 0, 0);
%end
curDir = pwd;
cd(dataPath);
norm_cols = @(m) m./sum(abs(m));

vecFileName = ['vecs_' num2str(n) '_' num2str(m) '_' num2str(numPoints) '_a' '.txt'];
cd('VecData')
%Allows for creation of up to 27 different data sets for a single number
%of datapoints
if exist(fullfile(cd, vecFileName), 'file') == 2
    validName = false;
    baseFileName = vecFileName(1:length(vecFileName) - 6);
    tempFileName = vecFileName;
    i = 0;
    while ~validName
        tempFileName = [ baseFileName '_' char('a' + i) '.txt'];
        if exist(fullfile(cd, tempFileName), 'file') ~= 2
            validName = true;
            vecFileName = tempFileName;
        end
        i = i + 1;
    end
    
    fprintf('Now using file %s...\n', vecFileName)
end

vecFileID = fopen(vecFileName, 'w');
cd('..')

cd('ConstData')
constFileName = ['const_' num2str(n) '_' num2str(m) '_' num2str(numPoints) '_a' '.txt'];
%See comment above
if exist(fullfile(cd, constFileName), 'file') == 2
    validName = false;
    baseFileName = constFileName(1:length(constFileName) - 6);
    tempFileName = constFileName;
    i = 0;
    while ~validName
        tempFileName = [ baseFileName '_' char('a' + i) '.txt'];
        if exist(fullfile(cd, tempFileName), 'file') ~= 2
            validName = true;
            constFileName = tempFileName;
        end
        i = i + 1;
    end
    
    fprintf('Now using file %s...\n', constFileName)
end

constFileID = fopen(constFileName, 'w');
cd('..')

printStr = '%.10f ';
printStr = repmat(printStr, 1, n * m - 1);
printStr = [printStr '%.10f\n'];

if(showBar == true)
    progressbar('Random Points');
end

for k = 1:numPoints
      f = rand_matr(n, m);
      [Q,~] = qr(f); U = Q(:,m + 1:n);         % U is the space f^\perp
      p_comp = MinProjCoor(U,Inf);         % computed projection constant of f^\perp
      fprintf(vecFileID, printStr, f);
      fprintf(constFileID, '%.10f\n', p_comp);
      if(showBar == true)
        progressbar(k/numPoints);
      end
end

fclose(vecFileID);
fclose(constFileID);

cd(curDir)

setpref('Internet', 'E_mail', 'ryanm@fourier.math.tamu.edu')
setpref('Internet', 'SMTP_Server', 'smtp-relay.tamu.edu')
sendmail('ryan_malthaner@tamu.edu', ['Data generation for ' vecFileName ' finished'])
end
