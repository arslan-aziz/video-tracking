%Input: Video directory
%       Optical flow output
%Output: Pos & dims of rectangular window around object in each frame
%Performs object tracking in each frame in three steps
%1) Segment object from background using optical flow-based segmentation
%2) Coarse template creation around object in foreground
%3) Refinement of tracking window using normalized template matching
%%
%Load video frames
file=dir('img');
file=file(3:end);
frames=length(file);
I=cell(frames,1);
for k = 1:frames
  I{k} = imread(fullfile('img', file(k).name));
end

I2=cellfun(@rgb2gray,I,'UniformOutput',false);
I2=im2double(cat(3,I2{:}));
dims=size(I2);

%Create template from first frame
%load('cfg.mat');
%ground=seq.gt_rect;
ground=readtable('groundtruth_rect.txt');
ground=table2array(ground);
start=ground(1,:);
yLength=start(1)+start(3);
xLength=start(2)+start(4);

temp=cell(1,frames);
temp{1}=I2(start(2):xLength,start(1):yLength,1);

%%
%Remove background through phase segmentation
%PERFORM IF BACKGROUND MOVES
% phaseMap=zeros(size(I2));
% for i=1:frames
%     phase=atan(V(:,:,i)./U(:,:,i));
%     phase=phase+pi;
%     phase=phase./max(max(phase));
%     phaseMap(:,:,i)=phase;
% end

%%
%Generate motion segmented mask
flowMask=zeros(size(flowMag));
for i=1:frames
    mask=flowMag(:,:,i);
    mask(mask>0.6)=1;
    mask(mask<0.6)=0;
    mask=imclose(mask,strel('square',10));
    comp=bwconncomp(mask);
    numPixels=cellfun(@numel,comp.PixelIdxList);
    indices=vertcat(comp.PixelIdxList{numPixels>200});
    if isempty(indices)
        [~,idx]=maxk(numPixels,2);
        indices=vertcat(comp.PixelIdxList{idx});
    end
    %[x,y]=ind2sub([size(flowMask,1) size(flowMask,2)],vertcat(comp.PixelIdxList{idx}));
    maskClean=zeros(size(mask));
    maskClean(indices)=1;
    flowMask(:,:,i)=maskClean;
end

%calculate proportion of height that is object
idx=find(flowMask(:,:,1));
[x0,y0]=ind2sub([size(flowMask,1),size(flowMask,2)],idx);
yProp=ground(1,4)/(max(x0)-min(x0));

%loop through frames to create coarse window
coarse=zeros(size(ground));
expCoarse=zeros(size(ground));
for i=2:frames-1
    idx=find(flowMask(:,:,i));
    [x,y]=ind2sub([size(flowMask,1),size(flowMask,2)],idx);
    yTop=min(x);
    yBot=floor((max(x)-min(x))*(yProp+0.1)+yTop);
    xLeft=min(y);
    if isempty(xLeft)
        continue;
    end
    xRight=max(y);
    coarse(i,1)=xLeft;
    coarse(i,2)=xRight;
    coarse(i,3)=yTop;
    coarse(i,4)=yBot;
    expCoarse(i,1)=coarse(i,1);
    expCoarse(i,2)=coarse(i,3);
    expCoarse(:,3)=coarse(:,2)-coarse(:,1);
    expCoarse(:,4)=coarse(:,4)-coarse(:,3);
end

%perform scaled template matching within bounds to refine window
exp=zeros(size(ground));
scale=0.8:0.05:1.2;
for i=2:frames-1
    i
    slice=double(I2(:,:,i));
    score=zeros(dims(1),dims(2),length(scale));
    for s=1:length(scale)
        %scale previous template to make new template
        tempS=imresize(temp{1},scale(s));
        %template size
        tS=size(tempS);
        %energy of template
        E_T=sum(sum(tempS.^2));
        
        for x=coarse(i,3):coarse(i,4)-tS(1)
            for y=coarse(i,1):coarse(i,2)-tS(2)
                comp=I2(x:x+tS(1)-1,...
                    y:y+tS(2)-1,i);
                %compute score based on correlation
                score(x,y,s)=sum(sum(comp.*tempS));
                %energy of windowed image
                E_BT=sum(sum(comp.^2));
                %normalize score to image and template energies
                score(x,y,s)=score(x,y,s)/sqrt(E_BT*E_T);
            end
        end
    end
    %find position--scale with best score (==1)
    [~,idx]=max(score(:));
    %find window in I2 again that contains template:used for comparison in 
    %next iteration of loop
    [posX,posY,sc]=ind2sub([size(score,1),size(score,2),size(score,3)],idx);
    exp(i,1)=posY;
    exp(i,2)=posX;
    exp(i,3)=scale(sc)*size(temp{1},2);
    exp(i,4)=scale(sc)*size(temp{1},1);
end
    
