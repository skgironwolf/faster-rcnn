%label_file = 'labels-KDOX-2011.csv';
label_file = 'labels.csv';
 
% Get struct array with all roosts
%   Note: second argument gives format specifiers
%     these are fairly universal. See help sprintf, fprintf
roosts = csv2struct(label_file, '%f%s%f%s%f%f%f%f%f%f%f%f%f%f');
nRoosts = numel(roosts); % number of roosts 
% fid=fopen('info.dat','w');
% %go through roosts and run through loss function (lf), then minimize lf
for i=1:nRoosts
docNode = com.mathworks.xml.XMLUtils.createDocument('annotation');
drn = docNode.getDocumentElement;

folder = docNode.createElement('folder');
folder.appendChild(docNode.createTextNode('CNNData'));
drn.appendChild(folder);

filename = docNode.createElement('filename');
filename.appendChild(docNode.createTextNode(strcat(int2str(i),'.jpg')));
drn.appendChild(filename);

source = docNode.createElement('source');

database = docNode.createElement('database');
database.appendChild(docNode.createTextNode(''));
source.appendChild(database);

annotation = docNode.createElement('annotation');
annotation.appendChild(docNode.createTextNode(''));
source.appendChild(annotation);

image = docNode.createElement('image');
image.appendChild(docNode.createTextNode(''));
source.appendChild(image);

s_flickerid = docNode.createElement('flickerid');
s_flickerid.appendChild(docNode.createTextNode(''));
source.appendChild(s_flickerid);

drn.appendChild(source);

owner = docNode.createElement('owner');

o_flickerid = docNode.createElement('flickerid');
o_flickerid.appendChild(docNode.createTextNode(''));
owner.appendChild(o_flickerid);

name = docNode.createElement('name');
name.appendChild(docNode.createTextNode('Saadia K. Gabriel'));
owner.appendChild(name);

drn.appendChild(owner);

size = docNode.createElement('size');

width = docNode.createElement('width');
width.appendChild(docNode.createTextNode('600'));
size.appendChild(width);

height = docNode.createElement('height');
height.appendChild(docNode.createTextNode('600'));
size.appendChild(height);

depth = docNode.createElement('depth');
depth.appendChild(docNode.createTextNode('3'));
size.appendChild(depth);

drn.appendChild(size);

segmented = docNode.createElement('segmented');
segmented.appendChild(docNode.createTextNode('0'));
drn.appendChild(segmented);

object = docNode.createElement('object');

object_name = docNode.createElement('name');
object_name.appendChild(docNode.createTextNode('roost'));
object.appendChild(object_name);

pose = docNode.createElement('pose');
pose.appendChild(docNode.createTextNode('Unspecified'));
object.appendChild(pose);

truncated = docNode.createElement('truncated');
truncated.appendChild(docNode.createTextNode('0'));
object.appendChild(truncated);

difficult = docNode.createElement('difficult');
difficult.appendChild(docNode.createTextNode('0'));
object.appendChild(difficult);

roo = roosts(i);
dataRoot = '/Users/saadiagabriel/Documents/stations';
fileName = sprintf('%s/%s/%04d/%02d/%02d/%s', dataRoot, ...
roo.station, roo.year, roo.month, roo.day, roo.filename);
radar = rsl2mat(fileName, roo.station); 
rmax = 150000;
dim = 600;
sweep = radar.dz.sweeps(1);
[Z,X,Y] = sweep2cart(sweep,rmax,dim);

x0 = min(X);
y0 = min(Y);
dx = mean(abs(diff(X)));   % meters per pixel
dy = mean(abs(diff(Y)));   % meters per pixel

roost_i = round((roo.x - x0)/dx) + 1; 
roost_j = round((max(y)-roo.y)/dy) + 1; 
roost_r = round(roo.r / dx);
minX = (roost_i-round(1.5*roost_r));
minY = (roost_j-round(1.5*roost_r));
box_d = 3*roost_r;

%save(strcat('bbox',int2str(i),'.mat'),'X','Y','radar');

bndbox = docNode.createElement('bndbox');

xmin = docNode.createElement('xmin');
xmin.appendChild(docNode.createTextNode(int2str(minX)));
bndbox.appendChild(xmin);

ymin = docNode.createElement('ymin');
ymin.appendChild(docNode.createTextNode(int2str(minY)));
bndbox.appendChild(ymin);

xmax = docNode.createElement('xmax');
xmax.appendChild(docNode.createTextNode(int2str(minX+box_d)));
bndbox.appendChild(xmax);

ymax = docNode.createElement('ymax');
ymax.appendChild(docNode.createTextNode(int2str(minY+box_d)));
bndbox.appendChild(ymax);

object.appendChild(bndbox);
drn.appendChild(object);

xmlFileName = [int2str(i),'.xml'];
xmlwrite(xmlFileName,docNode);
 end
% fclose(fid);