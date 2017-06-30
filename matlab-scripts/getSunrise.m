%Sample code:

%will need to parse the filename to get month, day, year
%extract lat/lon of station from radar struct
root = '/Users/saadiagabriel/Documents/stations/KDOX/2015/09/';
days = dir(root);
days_size = size(days,1);
station = 'KDOX';

for i = 1:days_size
    path = fullfile(strcat(root,days(i).name),'*.gz');
    files = dir(path);
    files = {files.name};
    for j = 1:size(files,2)
        radar_file = strcat(root,days(i).name,'/',files{j});
        radar = rsl2mat(radar_file,station);
        [rhr,rmin] = sunrise(radar.month, radar.day, radar.year, radar.lat, -radar.lon);
        rhr = mod(rhr,24);
        sunrise_t = datenum(radar.year, radar.month, radar.day, rhr, rmin, 0);
        scan_t = datenum(radar.year, radar.month, radar.day, radar.hour, radar.minute, radar.second);
        minutes_from_sunrise = (scan_t - sunrise_t)*(60*24);
        if(abs(minutes_from_sunrise) < 30)
            filename = strsplit(files{j},'.');
            filename = filename{1};
            SingleSample(filename,radar);
        end
    end
end


% radar_file = '/Users/saadiagabriel/Documents/stations/KDOX/2015/10/08/KDOX20151008_115900_V04.gz';
% station    = 'KDOX';
% radar      = rsl2mat(radar_file, station);
% [rhr,rmin] = sunrise(radar.month, radar.day, radar.year, radar.lat, -radar.lon);
% rhr = mod(rhr,24);
% sunrise_t = datenum(radar.year, radar.month, radar.day, rhr, rmin, 0);
% scan_t = datenum(radar.year, radar.month, radar.day, radar.hour, radar.minute, radar.second);
% minutes_from_sunrise = (scan_t - sunrise_t)*(60*24);