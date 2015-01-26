//
//  ExtractSIFT.cpp
//  invertedIndex
//
//  Created by xuhuaiyu on 14-12-4.
//  Copyright (c) 2014年 xuhuaiyu. All rights reserved.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>

using namespace cv;
using namespace std;

//此处定义了K-means聚类的中心数,注意定义的聚类中心数肯定是不能大于特征点的个数了
//本程序只针对img1进行了聚类，所以宏定义的种类数较少


void extractSIFTDescriptor(string);//提取SIFT特征
Mat k_means(Mat& , int );//k-means聚类
/**
 1.提取SIFT特征点
 2.获得所有图片的SIFT特征点的特征描述符矩阵（此处还需要写一个管理图片训练集的类）
 3.利用特征描述符矩阵进行K-means聚类，获得视觉单词表
 */

/**
 提取SIFT特征：
 
 dir:图片训练集的目录
 */
void extractSIFTDescriptor(string dir){
    cv::initModule_nonfree();//使用SIFT/SURF create之前，必须先initModule_<modulename>();
    
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
    
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SIFT" );
    if( detector.empty() || descriptorExtractor.empty() )
    {
        cout << "Can not create detector or descriptor extractor of given types" << endl;
        return ;
    }
    
    Mat allDescriptors ;//将所有图片的特征描述符存储在一起
    
    
    //------------------------------------------------------------------------------------------
    //下面需要写一个管理图片训练集的类
    /**
     遍历训练图片集文件夹，提取每张图片的SIFT特征点及特征点的特征描述符并合并在一起
     */
    Mat img;
    vector<KeyPoint> keypoints;
    detector->detect( img, keypoints );//特征点
    
    Mat descriptors;//特征点的特征描述符
    descriptorExtractor->compute( img, keypoints, descriptors );
    
    allDescriptors.push_back(descriptors);//将该图像所有特征点的特征描述符附加到总特征描述符矩阵的末尾
    
    //------------------------------------------------------------------------------------------
    
    
    
    
   //此时已经得到了allDescriptor，及所有特征点的特征描述符,下面该进行聚类和统计词频的操作
    int clusterNumber = 10;//k_means聚类中心数
    Mat center = k_means(allDescriptors, clusterNumber);//center矩阵存储视觉单词表
   
    
}
/**
 k-means聚类:
 
 allDescriptors:所有图片的特征点的特征描述符矩阵
 clusterNum:类别数
 
 return:聚类后的视觉单词表
 */
Mat k_means(Mat& allDescriptors, int clusterNum){
    
    BOWKMeansTrainer bowK(clusterNum,
                          cvTermCriteria (CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1),3,2);
    return bowK.cluster(allDescriptors);
}

