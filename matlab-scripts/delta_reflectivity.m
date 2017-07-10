label_file = 'labels.csv';
 
% Get struct array with all roosts
%   Note: second argument gives format specifiers
%     these are fairly universal. See help sprintf, fprintf
tic

% get roosts, sequence ids, and scan ids 
roosts = csv2struct(label_file, '%f%s%f%s%f%f%f%f%f%f%f%f%f%f');
nRoosts = numel(roosts); % number of roosts 
seqIDs = unique([roosts.sequence_id]);
scanIDs = unique([roosts.scan_id]); 

%roost instances that are part of each of the sequences 
info = cell(numel(seqIDs));

% dict for looking up delta reflectivity matrix relating to that scan 
c = containers.Map;

%works but inefficient, perhaps there is a way to vectorize? 
for i=1:numel(seqIDs)
    
    %all roost instances that are part of sequence 
    info(i) = {roosts([roosts.sequence_id] == seqIDs(i))}; 
    
    % for sequence id i, finds which scans contain roost instances that are part of that sequence 
    entry = [info{i}.scan_id];

    %reflectivity matrices for scans 
    t = cell(numel(entry));
    
    %for each scan 
    for j=1:numel(entry)
        %load pre-saved scan 
        t(j) = {load(strcat(int2str(entry(j)),'.mat'))}; 
        
        %calculate delta reflectivity 
        dr = zeros(600);
        if(j-1 ~= 0)
           t{j}.channel1(isnan(t{j}.channel1)) = 0;
           dr = t{j}.channel1 - t{j-1}.channel1;
        end
        c(int2str(entry(j))) = dr;
    end
end
toc