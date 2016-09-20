function extract_cnn_patches(video_names,param)

% create cache folders
cdirs = {};
if param.use_poses
cdirs={'patches_app','patches_resnet','patches_flow','patches_app/left_hand','patches_resnet/left_hand','patches_flow/left_hand', ...
    'patches_app/right_hand','patches_resnet/right_hand','patches_flow/right_hand','patches_app/upper_body','patches_resnet/upper_body','patches_flow/upper_body', ...
    'patches_app/full_body','patches_resnet/full_body','patches_flow/full_body','patches_app/full_image','patches_resnet/full_image','patches_flow/full_image'};
else
    cdirs={ 'patches_app/top_left','patches_flow/top_left','patches_app/top_right','patches_flow/top_right',...
    'patches_app/bottom_left','patches_flow/bottom_left','patches_app/bottom_right','patches_flow/bottom_right', ...
     'patches_app/center','patches_flow/center','patches_app/full_image','patches_flow/full_image'};
end
for d=1:length(cdirs)
    dname=sprintf('%s/%s',param.cachepath,cdirs{d});
    if ~exist(dname,'dir'); mkdir(dname) ; end
end

fprintf('\n------ Compute CNN patches ------\n')

parfor vi = 1:length(video_names)
    fprintf('extract patches .. : %d out of %d videos\n',vi,length(video_names))
    
    % get image list in the current video
    vidname=video_names{vi} ;
    images=dir(sprintf('%s/%s/*%s',param.impath,vidname,param.imext));
    images = {images.name};
    
     jointFile = sprintf('%s/%s/joint_positions',param.jointpath,vidname);
    % get video joint positions and human scales
%     if ~exist(jointFile,'file'); continue ; end ;
    if param.use_poses	
    	positions=load(jointFile) ;
    	scale=positions.scale ;
    	positions=positions.pos_img ;
    end
    suf={'app','flow','resnet'} ;
    imdirs = {param.impath,sprintf('%s/OF',param.cachepath),param.impath};
    
    for i=1:3 % appearance and flow 
        imdirpath = imdirs{i};
        
        net=param.(sprintf('net_%s',suf{i}));
        
        %for idim=1:min(length(images),length(positions))
	for idim=1:length(images)
            if exist(sprintf('%s/full_image/%s_im%05d.jpg',param.cachepath,vidname,idim),'file')
                continue;
            end
            if strcmp(suf{i},'resnet') == 1
                imgSize = net.meta.normalization.imageSize(1:2);
            else    
                imgSize = net.normalization.imageSize(1:2);
            end
      
            % get image
            if i~=2 % appearance
                impath = sprintf('%s/%s/%s',imdirpath,vidname,images{idim}) ;
                
            else % flow
                [~,iname,~]=fileparts(images{idim});
                impath = sprintf('%s/%s/%s.jpg',imdirpath,vidname,iname) ; % flow has been previously saved in JPG
                if ~exist(impath,'file'); continue ; end ; % flow was not computed (see compute_OF.m for info)
            end
            im = imread(impath);
            [height,width,~] = size(im)
            % get part boxes
            
            
           
	if param.use_poses 
           
           % part CNN (fill missing part before resizing)
            sc=scale(idim); lside=param.lside*sc ;
            
            % left hand
            lhand = get_box_and_fill(positions(:,param.lhandposition,idim)-lside,positions(:,param.lhandposition,idim)+lside,im);
            lhand = imresize(lhand, imgSize) ;
            imwrite(lhand,sprintf('%s/patches_%s/left_hand/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));

            % right right
            rhand = get_box_and_fill(positions(:,param.rhandposition,idim)-lside,positions(:,param.rhandposition,idim)+lside,im);
            rhand = imresize(rhand, imgSize) ;
            imwrite(rhand,sprintf('%s/patches_%s/right_hand/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));

            % upper body
            sc=scale(idim); lside=3/4*param.lside*sc ;
            upbody = get_box_and_fill(min(positions(:,param.upbodypositions,idim),[],2)-lside,max(positions(:,param.upbodypositions,idim),[],2)+lside,im);
            upbody = imresize(upbody, imgSize) ;
            imwrite(upbody,sprintf('%s/patches_%s/upper_body/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));

            % full body
            fullbody = get_box_and_fill(min(positions(:,:,idim),[],2)-lside,max(positions(:,:,idim),[],2)+lside,im);
            fullbody = imresize(fullbody, imgSize) ;
            imwrite(fullbody,sprintf('%s/patches_%s/full_body/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));            
        else    
	    % top Left
	    topLeft = get_box_and_fill([1,1],[width/2,height/2],im);
            topLeft = imresize(topLeft, imgSize);
            imwrite(topLeft,sprintf('%s/patches_%s/top_left/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));

	    % top right	
            topRight = get_box_and_fill([width/2,1],[width,height/2],im);
            topRight = imresize(topRight, imgSize);
            imwrite(topRight,sprintf('%s/patches_%s/top_right/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
		
            % Bottom Left
	    bottomLeft = get_box_and_fill([1,height/2],[width/2,height],im);	
            bottomLeft = imresize(bottomLeft, imgSize);
            imwrite(bottomLeft,sprintf('%s/patches_%s/bottom_left/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
 	    
            % Bottom Right
            bottomRight = get_box_and_fill([width/2,height/2],[width,height],im);	
            bottomRight = imresize(bottomRight, imgSize);
            imwrite(bottomRight,sprintf('%s/patches_%s/bottom_right/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));    
        
            % center
            center = get_box_and_fill([width/4,height/4],[width - width*0.25,height - height*0.25],im);
            center = imresize(center, imgSize);
            imwrite(center,sprintf('%s/patches_%s/center/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
       end	    
            % full image CNNf (just resize frame)
            fullim = imresize(im, imgSize) ;
            imwrite(fullim,sprintf('%s/patches_%s/full_image/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
	    
        end
    end
end
