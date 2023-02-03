%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created by Hny on 2023.02.02
% FeatureMapVisualization 特征图可视化类
% 提供了一个用于特征图可视化的类及其内置方法函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 使用
%**************************************************************************
% 特征图可视化类
% 属性：
% （1）model：网络模型
% （2）maxItem：网络层数
% （3）className：网络可所属的类
% （4）callBackTips：回调函数可选的提示
% （5）Layers：解析网络各层
% （6）inputSize：网络输入层的尺寸
% （7）featureMaps：各层计算得到的特征图数据
% （8）cmap：可视化色阶
% 方法函数：
% （1）FeatureMapVisualization：构造函数，初始化类内的各参数。
%   （a）model：必要参数。传入模型
%   （b）isShow：缺省参数（默认值为false）。用于确定在初始化后是否分析网络
% （2）show：可视化函数
%   （a）featureMap：必要参数。特征图数据，维度为w*h*c
%   （b）k：必要参数。需要可视化的网络的层数
%   （c）type：必要参数。保存可视化结果的文件类型，fig为最优
%   （d）Title：必要参数。figgure图的标题
%   （e）isSave：必要参数。用于确定是否保存fig图
% （3）visualization：网络单层可视化函数
%   （a）image：必要参数。输入图像数据
%   （b）k：缺省参数。需要可视化的网络的层数
%   （c）type：缺省参数。保存可视化结果的文件类型，fig为最优
%   （d）resolution：缺省参数。可视化结果的分辨率，通过上采样获取
%   （e）isSave：缺省参数。用于确定是否保存fig图
% （4）visualizationEveryLayer：网络所有层可视化函数
%   （a）image：必要参数。输入图像数据
%   （b）type：缺省参数。保存可视化结果的文件类型，fig为最优
%   （c）resolution：缺省参数。可视化结果的分辨率，通过上采样获取
%   （d）isSave：缺省参数。用于确定是否保存fig图
% （5）callBack：回调函数
%   （a）type：报错类型
%**************************************************************************

%% 特征图可视化类
classdef FeatureMapVisualization
    properties
        model = [];
        maxItem = 0;
        className = ["SeriesNetwork","DAGNetwork","nnet.cnn.LayerGraph"];
        class = ' ';
        callBackTips = [...
            "'Error':The input model does not conform to the standard class('SeriesNetwork','DAGNetwork','nnet.cnn.LayerGraph')."; ...
            "'Error':The data type of the input model must be a list."];
        Layers = [];
        inputSize = [];
        featureMaps = struct;
        cmap = [];
    end
    
    methods
        function obj = FeatureMapVisualization(model,varargin)
            ip = inputParser;
            addParameter(ip,'isShow',false);
            parse(ip,varargin{:});
            isShow = ip.Results.isShow;

            obj.model = model;
            obj.class = class(obj.model);
            if ~ismember(obj.class,obj.className); obj.callBack(1); end
            obj.Layers = obj.model.Layers;
            obj.maxItem = length(obj.Layers);
            obj.inputSize = obj.Layers(1).InputSize;
            if isShow; analyzeNetwork(obj.model); end
        end
        
        function show(obj,featureMap,k,type,Title,isSave)
            makedir('featureMap');
            savepath = "featureMap/layer-"+num2str(k)+"."+type;
            mapSize = size(featureMap);
            figure('Position',[100,200,600,600],'Name',Title)
            h = ceil(sqrt(mapSize(3))); w = ceil(mapSize(3)/h);
            tight_subplot(h,w,[0,0],[0,0],[0,0],featureMap,obj.cmap);
            if isSave; savefig(savepath); else; pause(0.01);end
        end

        function featureMap = visualization(obj,image,varargin)
            ip = inputParser;
            addParameter(ip,'k',1); addParameter(ip,'type','fig'); addParameter(ip,'resolution',[]);addParameter(ip,'cmap',[]);addParameter(ip,'isSave',false);
            parse(ip,varargin{:});
            k = ip.Results.k; type = ip.Results.type; resolution = ip.Results.resolution; obj.cmap = ip.Results.cmap; isSave = ip.Results.isSave;
            layerName = obj.Layers(k).Name;
            input = imresize(image,obj.inputSize(1:2));
            featureMap = activations(obj.model,input,layerName);
            if ~isempty(resolution);  featureMap = imresize(featureMap,resolution); end
            obj.show(featureMap,k,type,layerName,isSave);
            eval("obj.featureMaps."+obj.Layers(k).Name+"=featureMap;");
        end

        function featureMaps = visualizationEveryLayer(obj,image,varargin)
            ip = inputParser;
            addParameter(ip,'type','fig'); addParameter(ip,'resolution',[]);addParameter(ip,'cmap',[]);addParameter(ip,'isSave',false);
            parse(ip,varargin{:});
            type = ip.Results.type; resolution = ip.Results.resolution; obj.cmap = ip.Results.cmap;isSave = ip.Results.isSave;
            for i = 1:obj.maxItem
                fprintf('[%d/%d] layerName:%s\n',i,obj.maxItem,obj.Layers(i).Name);
                obj.visualization(image,k=i,type=type,resolution=resolution,isSave=isSave);
            end
            featureMaps = obj.featureMaps;
        end

        function callBack(obj,type)
            disp(obj.callBackTips(type));
        end
    end
end


%% 
% 创建空文件夹
function makedir(folder)
    if ~exist(folder,'dir'); mkdir(folder); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pekka Kumpulainen (2023). tight_subplot(Nh, Nw, gap, marg_h, marg_w) 
% (https://www.mathworks.com/matlabcentral/fileexchange/27991-tight_subplot-nh-nw-gap-marg_h-marg_w), 
% MATLAB Central File Exchange. 检索来源 2023/2/2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tight_subplot(Nh, Nw, gap, marg_h, marg_w,featureMap,cmap)
    hWaitbar = waitbar(0, 'Waiting for imshowing...', 'CreateCancelBtn', 'delete(gcbf)','name','Progress bar');
    set(hWaitbar, 'Color', [0.9, 0.9, 0.9]);
    if numel(gap)==1; gap = [gap gap]; end
    if numel(marg_w)==1; marg_w = [marg_w marg_w]; end
    if numel(marg_h)==1; marg_h = [marg_h marg_h]; end
    
    axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh; 
    axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;
    
    py = 1-marg_h(2)-axh; 
    
    % ha = zeros(Nh*Nw,1);
    ii = 0; ha = 1:Nw*Nh; [~,~,s] = size(featureMap);
    for ih = 1:Nh
        px = marg_w(1);
        for ix = 1:Nw
            ii = ii+1;
            if ii > s
                close(hWaitbar);
                return
            end
            ha(ii) = axes('Units','normalized', ...
                'Position',[px py axw axh], ...
                'XTickLabel','', ...
                'YTickLabel','');
            px = px+axw+gap(2);
            imshow(featureMap(:,:,ii),cmap);
            if ishandle(hWaitbar)
                percent = ii/s;
                waitbar(percent,hWaitbar,"Waiting for imshowing..."+num2str(round(percent,4)*100)+"%");
            end
        end
        py = py-axh-gap(1);
    end
    close(hWaitbar);
end