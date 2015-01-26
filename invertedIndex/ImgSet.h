//
//  ImgSet.h
//  invertedIndex
//
//  Created by xuhuaiyu on 14-12-4.
//  Copyright (c) 2014年 xuhuaiyu. All rights reserved.
//


/**
 该类用于进行图像管理
 */
#ifndef __invertedIndex__ImgSet__
#define __invertedIndex__ImgSet__


#include <iostream>
#include <vector>

#endif /* defined(__invertedIndex__ImgSet__) */

using namespace std;
class ImgSet{
public:
    
    string getImgName(int idx){
        return imgs.at(idx);
    }
    void setImgName(string inImgName){
        imgName = inImgName;
    }
    int getTotalImgNum(){
        return imgNum;
    }
    void setImgsNum(int imgsNum){
        imgNum = imgsNum;
    }
    void loadImgsFromDir(string);
    
private:
    string imgName;
    int imgNum = 0;
    typedef vector <string> IMG_SET;
    IMG_SET imgs;
};
