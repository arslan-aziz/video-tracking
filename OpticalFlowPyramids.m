%Load video frames
%Lucas-Kanade Optical Flow implementation
%Use gaussian pyramids to refine flow across multiple scales
%Input: Video directory
%Output: 3D matrix containing u and v optical flow components at each frame
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

%%
%Separated Prewitt operators
%ker_x=[-1,1;-1,1]; 
%ker_y=[-1,-1;1,1];
ker_x1=[1 1 1]';
ker_x2=[-1 0 1];
ker_y1=[-1 0 1]';
ker_y2=[1 1 1];
%ker_t=[1 1;1 1];

flowMag=zeros(size(I2));
U=zeros(size(I2));
V=zeros(size(I2));
thresh=0.02;
for t=1:frames-1
    %images to find optical flow
    t1=I2(:,:,t);
    t2=I2(:,:,t+1);
    
    t1P=generatePyramid(t1,7);
    t2P=generatePyramid(t2,7);
    
    u=zeros(floor(size(t1P{end})/2));
    v=zeros(floor(size(t1P{end})/2));
    for k=length(t1P)-2:-1:1
        uP=2*imresize(u,size(t1P{k}));
        vP=2*imresize(v,size(t1P{k}));
        uP(isnan(uP))=0;
        vP(isnan(vP))=0;
        u=zeros(size(uP));
        v=zeros(size(vP));
        
        %warp image at level k, time t
        [cx2,cy2]=meshgrid(1:size(t2P{k},2),1:size(t2P{k},1));
        t2Warp=interp2(cx2,cy2,t2P{k},cx2+vP,cy2+uP);
        t2Warp(isnan(t2Warp))=0;
        %apply ker_x to first image to get dI/dx
        %fx=conv2(t1P{k},ker_x,'same');
        fx=conv2(t1P{k},ker_x1,'same');
        fx=conv2(fx,ker_x2,'same');
        %apply ker_y to first image to get dI/dy
        %fy=conv2(t1P{k},ker_y,'same');
        fy=conv2(t1P{k},ker_y1,'same');
        fy=conv2(fy,ker_y2,'same');
        %apply ker_t to first image and neg. of second to get dI/dt
        %ft=conv2(t1P{k},ker_t,'same')+conv2(t2Warp,-1*ker_t,'same');
        ft=t2Warp-t1P{k};
        
        %loop through all points, extract window, and fit OF equation
        %window size here is 5x5
        for i=3:size(t1P{k},1)-3
            for j=3:size(t1P{k},2)-3
                Ix=fx(i-2:i+2,j-2:j+2);
                Iy=fy(i-2:i+2,j-2:j+2);
                It=ft(i-2:i+2,j-2:j+2);
                %optical flow eq: fx*u+fy*v+ft=0
                A=[Ix(:) Iy(:)];
                b=It(:);
                %***THRESHOLD EIGENVALUES***
                square=A'*A;
                e=eig(square);
                if any(e<thresh)
                    flow=[0 0];
                else
                    flow=square\(A'*b);
                end
                %add delta flow for pyramid level k
                u(i,j)=uP(i,j)+flow(1);
                v(i,j)=vP(i,j)+flow(2);
            end
        end
    end
    
    U(:,:,t)=u;
    V(:,:,t)=v;
    flowMag(:,:,t)=sqrt(u.^2+v.^2);
    t
end

%%
%test pyramid generation
test=I2(:,:,5);
lim=7;
t1P=generatePyramid(test,lim);
%%
%Generate image pyramids
function gaussPyramid = generatePyramid(image,lim)
    bound=min([size(image,1),size(image,2)]);
    levels=floor(log(bound/lim)/log(2));
    gaussPyramid=cell(1,levels);
    gaussPyramid{1}=image;
    
    a=0.375;
    %gauss shape from MATLAB website
    gauss=[0.25-a/2,0.25,a,0.25,0.25-a/2];
    for i=2:levels+1
        %gaussian blur rows then cols
        imageR=conv2(gaussPyramid{i-1},gauss,'same');
        imageR=conv2(imageR,gauss','same');
        %sample every other col/row
        imageR=imageR(2:2:end,2:2:end);
        gaussPyramid{i}=imageR;
    end
end


