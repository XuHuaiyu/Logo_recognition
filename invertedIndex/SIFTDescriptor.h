//
//  SIFTDescriptor.h
//  invertedIndex
//
//  Created by xuhuaiyu on 14-12-4.
//  Copyright (c) 2014年 xuhuaiyu. All rights reserved.
//

#ifndef __invertedIndex__SIFTDescriptor__
#define __invertedIndex__SIFTDescriptor__

#include <iostream>
#include <vector>
#include "opencv2/features2d/features2d.hpp"

#endif /* defined(__invertedIndex__SIFTDescriptor__) */

using namespace cv;
using namespace std;
class SIFTDiscriptor
{
public:
    int GetInterestPointNumber()
    {
        return interestPointNumber;
    }
    struct vector<KeyPoint> getKeyPoints()
    {
        return keypoints;
    }
    struct Mat getDescriptor(){
        return descriptors;
    }
    void setImgName(const std::string &strImgName)
    {
        m_strInputImgName = strImgName;
    }
    string getImgName(){
        return m_strInputImgName;
    }
    void setKeyPoints(vector<KeyPoint> inKeypoints){
        keypoints = inKeypoints;
    }
    void setDescriptor(Mat inDescriptor){
        descriptors = inDescriptor;
    }
    void setIPNum(int IPNum){
        interestPointNumber = IPNum;
    }
    //    int CalculateSIFT();
public:
    SIFTDiscriptor(const std::string &strImgName);
    SIFTDiscriptor()
    {
        interestPointNumber = 0;
        //        m_pFeatureArray = NULL;
    }
    //    ~CSIFTDiscriptor();
private:
    string m_strInputImgName;  //图像名
    int interestPointNumber;     //特征点数
    vector<KeyPoint> keypoints;     //特征点
    Mat descriptors;                //特征描述符
};