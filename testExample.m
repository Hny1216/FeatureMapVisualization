%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created by Hny on 2023.02.02
% 测试例程
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 初始化配置
close all; clc;clear;
input = imread("butterfly.jpg");   % 加载图片
a = alexnet;    % 载入模型 
Fmv = FeatureMapVisualization(a,isShow=true); % 初始化

%% 测试
% 单层测试：可传参数有：
% 必要参数：input(图片数据)
% 缺省参数：k(层数)，cmap(色阶)，resolution(分辨率)，type(文件类型)，isSave(保存与否)
Fmv.visualization(input,k=2,cmap=hsv,resolution=[256,256]);


% 多层测试：
layerIndex = [1:10,12];
for i = 1:length(layerIndex)
    Fmv.visualization(input,k=i,isSave=false);
end


% 全网络测试：可传参数有：
% 必要参数：input(图片数据)
% 缺省参数：cmap(色阶)，resolution(分辨率)，type(文件类型)，isSave(保存与否)
Fmv.visualizationEveryLayer(input,cmap=gray,isSave=false);
