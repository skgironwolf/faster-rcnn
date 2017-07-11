label_file = 'labels.csv';
 
% Get struct array with all roosts
%   Note: second argument gives format specifiers
%     these are fairly universal. See help sprintf, fprintf
tic
roosts = csv2struct(label_file, '%f%s%f%s%f%f%f%f%f%f%f%f%f%f');
nRoosts = numel(roosts); % number of roosts 
scanIDs = unique([roosts.scan_id]); % number of unique radar scans 
min = -5;
max = 35;
minVR = -20;
maxVR = 20;

for i=1:numel(scanIDs)
     get_roosts = roosts([roosts.scan_id] == scanIDs(i));
     roo = get_roosts(1)
     dataRoot = '/Users/saadiagabriel/Documents/stations';
     fileName = sprintf('%s/%s/%04d/%02d/%02d/%s', dataRoot, ...
         roo.station, roo.year, roo.month, roo.day, roo.filename);
     radar = rsl2mat(fileName, roo.station); 
     rmax = 150000;
     dim = 600;
 
     %reflectivity at first elevation 
     sweep1 = radar.dz.sweeps(1);
     %radial velocity at first elevation 
     sweep3 = radar.vr.sweeps(1);
 
     %reflectivity at second elevation 
     if strcmp(radar.station,'KDOX')
         sweep2 = radar.dz.sweeps(2);
     else
         sweep2 = radar.dz.sweeps(3);
     end
     channel1 = sweep2cart(sweep1, rmax, dim);
     channel2 = sweep2cart(sweep2, rmax, dim);
     channel3 = sweep2cart(sweep3, rmax, dim);

     %scale values 
     channel1 = (channel1 - min)./(max-min);
     channel1(isnan(channel1)) = 0;
     channel2 = (channel2 - min)./(max-min);
     channel2(isnan(channel2)) = 0;
     channel3 = (channel3 - minVR)./(maxVR-minVR);
     channel3(isnan(channel3)) = 0;

     %render and save image
     image = zeros(600,600,3);
     image(:,:,1) = channel1;
     image(:,:,2) = channel2;
     image(:,:,3) = channel3;
     imwrite(image,strcat(int2str(scanIDs(i)),'.jpg'));

    %This line is probably not needed if using AWS 
    %save(strcat(int2str(scanIDs(i)),'.mat'),'channel1','channel2','channel3')
end
toc