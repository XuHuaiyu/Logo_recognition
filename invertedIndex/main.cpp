//
//  main.cpp
//  invertedIndex
//
//  Created by xuhuaiyu on 14-12-3.
//  Copyright (c) 2014年 xuhuaiyu. All rights reserved.
//

/**
 具体步骤：
 
 训练过程：
 1.加载训练集图片
 2.从每幅图片提取keypoint
 3.compute每个keypoint对应的descriptor
 4.k-means聚类获得聚类中心（即视觉单词表）
 5.利用视觉单词表统计训练集中每幅图片的词频向量
 6.利用得到的词频向量建立倒排索引
 
 测试过程：
 1.利用训练过程中得到的视觉单词表统计图片的词频向量
 2.利用词频向量在倒排索引中进行搜索，获得所有相应图片的相似度
 
 */

/*
 
 待优化部分：
 防止程序中途自己崩溃，需要对每一个大步骤的结果进行存储，以实现自行的恢复
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/flann/flann.hpp"   //包含分层kmeans方法的头文件

#include "SIFTDescriptor.h"
#include "ImgSet.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include <list>
#include <fstream>
using namespace cv;
using namespace std;
/*
 
 Kmeans之后构建单词表
 统计词频之后构建倒排列表
 
 clusterNumber表示聚类之后视觉单词表的大小，即倒排索引中单词表的大小
 */

//Mat extractSIFTDescriptor(string, SIFTDiscriptor*,ImgSet,vector<vector<KeyPoint>>&);//提取SIFT特征
//Mat extractSIFTDescriptor(string dir,SIFTDiscriptor* imgsDiscriptor,ImgSet imgSet, struct imgInfo* imgs);
//Mat extractSIFTDescriptor(string dir,SIFTDiscriptor* imgsDiscriptor,ImgSet imgSet,list<struct imgInfo>& imgs);
Mat extractSIFTDescriptor(string dir, ImgSet imgSet,list<struct imgInfo>& imgs);

Mat k_means(Mat& , int );//k-means聚类
Mat h_kmeans(Mat&, int);//hierarchical k_means
//struct wordsFreqNode* countWordsFreq(Mat, SIFTDiscriptor*);//计算词频
Mat countWordsFreq(Mat&, vector<KeyPoint> , vector<vector<int>>*, BOWImgDescriptorExtractor);//计算一幅图片的词频
void buildWordsList(int, struct visualWord* );//构建视觉单词表
void buildInvertedIndex(struct imgInfo*,  int,  int, struct visualWord*);//构建倒排列表
//int matchImg(int*, struct visualWord);//相似度匹配
vector<int> matchImg(string );//匹配图片
void imshowMany(const std::string& _winName, const vector<Mat>& _imgs);//一个窗口显示多幅图片
double cosSimlarity(int*, int*, int);//计算余弦相似度
int wordsFreqIntoFile(struct imgInfo*);
int invertedListIntoFile(struct visualWord*);


int trainingImgNumber = 0;//图片训练集的图片数
const string trainingImgDir = "/Users/xuhuaiyu/Development/trainingImgs";//图片训练集目录
const int clusterNumber = 100000;//K-means聚类的K值


struct imgInfo{//图片词频节点
    string imgPath;
    vector<KeyPoint> keypoint;//关键点
    Mat descriptors;//描述子
    Mat histogram;//词频

};
struct visualWord{//视觉单词表结构体
    //    string imgPath;//该图片对应的路径
    int imgAmt = 0;//记录该单词指向的图片数
    struct invertedTableNode* next;
};

struct invertedTableNode{//倒排列表的节点结构体
    int picId = -1;//图片编号
    //非零的词频值
    struct invertedTableNode* next;
};




/**
 读取视觉单词表的代码
 Mat dictionary;
 FileStorage fs("/Users/xuhuaiyu/Development/trainingResults/dictionary.yml", FileStorage::READ);
 fs["vocabulary"] >> dictionary;
 fs.release();
 */


int main(int argc, const char * argv[]) {
    
    
    cout<<"  请输入选择: "<<endl;
    cout<<"1.构建视觉单词表"<<endl;
    cout<<"2.统计训练集词频"<<endl;
    cout<<"3.建立倒排索引"<<endl;
    cout<<"4.测试"<<endl;
    char option;
    cin>>option;
    
    if( option == '1'  ){//
        //SIFTDiscriptor* imgsDiscriptor ;
        ImgSet imgSet ;
        
        cout<<"<开始加载训练集图片>"<<endl;
        imgSet.loadImgsFromDir(trainingImgDir) ;//加载训练集中的所有图片
        cout <<"<加载训练集图片完毕>"<<endl;
        trainingImgNumber = imgSet.getTotalImgNum();
        cout << "<共有图片" << trainingImgNumber <<"张>"<<endl;
       // imgsDiscriptor = new SIFTDiscriptor[trainingImgNumber];
        
       
        list<struct imgInfo> imgs;// 用以存储训练集所有图像的相关信息
        //vector<vector<KeyPoint>> keypoints;//用来存储每幅图片的关键点
        cout << "< SIFT特征提取 >" << endl;
        Mat allDescriptors = extractSIFTDescriptor(trainingImgDir, imgSet, imgs);//提取SIFT特征，allDescriptors记录所有图片的特征点的特征描述符
        cout<<"<提取SIFT特征完毕>"<<endl;
        
        cout<<"-----------------------------------"<<endl<<endl;
        
        cout<<endl<<"<共提取SIFT特征"<<allDescriptors.rows<<"个>"<<endl;
        
        
        
        cout << "< K-means聚类 >" << endl;
        Mat center = h_kmeans(allDescriptors, clusterNumber);//center矩阵存储视觉单词表
        cout << "< K-means聚类完毕 >" << endl;
        
        cout<<"-----------------------------------"<<endl<<endl;
        allDescriptors.release();//释放矩阵空间
        
        cout << " <dictionary 写入文件。。 > " << endl;
        FileStorage fs("/Users/xuhuaiyu/Development/trainingResults/dictionary_test.yml", FileStorage::WRITE);
        fs << "vocabulary" << center;
        fs.release();
        cout << " <dictionary 写入文件完毕 > " << endl;
        
        
        cout << "< 统计词频 >" << endl;
        cout<< "< 词频统计初始化操作 >" << endl;
        

        Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT"); //引号里面修改特征种类。
        Ptr<DescriptorMatcher>  matcher = DescriptorMatcher::create("BruteForce"); //引号里面修改匹配类型;
        BOWImgDescriptorExtractor bowDE(extractor, matcher);//前面两个定义是为了方便初始化类的定义，在BOW图像特征定义完成后，便可以对每一副图片提取BOW的特征。
        
        bowDE.setVocabulary(center); //设置单词表，center是通过前面聚类得到的聚类中心（即视觉单词表）；
        
        cout<<" < 开始词频统计 > " << endl;
        
        list<struct imgInfo>::iterator  itor;//构造list的迭代器，准备对每张图片进行遍历
        itor = imgs.begin();

        
        vector<vector<int>> pointIdxsOfClusters;//Indices of keypoints that belong to the cluster.
        
        for(int i = 0; i<trainingImgNumber && itor != imgs.end(); itor++,i++){
            string imgName = imgSet.getImgName(i);
            cout<<"<第"<<i+1<<"张：图片名："<<imgName<<">"<<endl;
            Mat img = imread(imgName);
            itor->histogram = countWordsFreq(img, itor->keypoint, &pointIdxsOfClusters, bowDE);
        }
        cout << "< 统计词频完毕 >" << endl;
        
        
        //词频写入文件(即TF)
        cout<<"<词频写入文件>"<<endl;
        Mat wordsFreq;//记录所有图片的直方图，第i行为第i张图片对应的直方图
        itor = imgs.begin();
        for(;itor != imgs.end(); itor++){
            wordsFreq.push_back(itor->histogram);
        }
        
        fs.open("/Users/xuhuaiyu/Development/trainingResults/TF.yml", FileStorage::WRITE);
        fs << "TF" << wordsFreq;
        fs.release();
        cout << " <TF 写入文件完毕 > " << endl;
        
        cout<<"-----------------------------------"<<endl<<endl;
        
        
        
        vector<vector<int>> invertedList(clusterNumber);//倒排索引，单词表的个数为聚类中心的个数
        //此处不知道内存空间是否够用
        vector<int> idfMat(clusterNumber);//记录每个视觉单词的idf
        cout<<" < 构建倒排列表。。。 > "<<endl;
        //列优先遍历词频文件，构造倒排列表
        for (int i = 0; i < wordsFreq.cols; i++){//i为列号，即第i个视觉中心
            int count = 0;
            for(int j = 0; j < wordsFreq.rows; j++){//j为行号，即第j张图片
                if(wordsFreq.at<float>(j, i)){
                    invertedList[i].push_back(j);
                    count++;//临时变量，用来记录含有该聚类中心的图片数
                }
            }
            idfMat[i] = trainingImgNumber*1.0/count;
        }
        cout<<" < 构建倒排列表完成 > "<<endl;

        
        //列优先遍历词频向量矩阵，计算每张图片的TF-IDF向量
        cout<<"< 计算每幅图片TF-IDF向量 >"<<endl;
        Mat tf_idf(wordsFreq.rows,wordsFreq.cols,CV_32F);//初始化tf_idf矩阵
        for(int i = 0 ; i < wordsFreq.cols; i++){//i为列号，即第i个视觉中心
            for(int j = 0 ; j < wordsFreq.rows; j++){//j为行号，即第j张图片
                tf_idf.at<float>( j, i ) = wordsFreq.at<float>(j,i) * idfMat[i];
            }
        }
        cout<<"< TF-IDF向量构建完毕 >"<<endl;
        
        cout<<"< 训练集图片TF-IDF向量写入文件。。。 >"<<endl;
        fs.open("/Users/xuhuaiyu/Development/trainingResults/TF-IDF.yml", FileStorage::WRITE);
        fs << "TF-IDF" << tf_idf;
        fs.release();
        cout << " <TF-IDF 写入文件完毕 > " << endl;

    }
    
    
    
    else if(option == '2'){//调用视觉单词表统计训练集中每幅图片的词频
        
        
        Mat dictionary;
        FileStorage fs("/Users/xuhuaiyu/Development/trainingResults/dictionary_test.yml", FileStorage::READ);
        fs["vocabulary"] >> dictionary;
        fs.release();
        cout << "< 统计词频 >" << endl;
        //wordsFreq二维数组用来记录词频
        cout<< "< 词频统计初始化操作 >" << endl;
        
        Ptr<FeatureDetector> detector(new SiftFeatureDetector());
        Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT"); //引号里面修改特征种类。
        Ptr<DescriptorMatcher>  matcher = DescriptorMatcher::create("BruteForce"); //引号里面修改匹配类型;
        BOWImgDescriptorExtractor bowDE(extractor, matcher);//前面两个定义是为了方便初始化类的定义，在BOW图像特征定义完成后，便可以对每一副图片提取BOW的特征。
        
        bowDE.setVocabulary(dictionary); //设置单词表，center是通过前面聚类得到的聚类中心（即视觉单词表）；

        
        ImgSet imgSet ;
        
        cout<<"<开始加载训练集图片>"<<endl;
        imgSet.loadImgsFromDir(trainingImgDir) ;//加载训练集中的所有图片
        cout <<"<加载训练集图片完毕>"<<endl;
        trainingImgNumber = imgSet.getTotalImgNum();
        
        cout<<" < 开始词频统计 > " << endl;
        vector<KeyPoint> keypoints;
        for(int i = 0 ; i < trainingImgNumber; i++){//遍历每张图片进行词频统计
            
            string imgName = imgSet.getImgName(i);
            cout<<"<第"<<i+1<<"张：图片名："<<imgName<<">"<<endl;
            Mat img = imread(imgName);
            detector->detect(img,keypoints);
            countWordsFreq(img, keypoints, NULL, bowDE);
        }
        
        cout << "< 统计词频完毕 >" << endl;
        cout<<"-----------------------------------"<<endl<<endl;

    }
    
    else if(option == '3'){//建立倒排索引,统计每个视觉词的IDF并计算
        Mat wordsFreq;
        FileStorage fs("/Users/xuhuaiyu/Development/trainingResults/TF.yml", FileStorage::READ);//读入词频文件
        fs["TF"] >> wordsFreq;
        fs.release();

        vector<vector<int>> invertedList(clusterNumber);//单词表的个数为聚类中心的个数
        //此处不知道内存空间是否够用
        
        Mat idfMat(1,clusterNumber,CV_32F);//行向量，记录每个视觉词的idf
       // vector<float> idfMat(clusterNumber);//记录每个视觉单词的idf
        cout<<" < 构建倒排列表。。。 > "<<endl;
        //列优先遍历词频文件，构造倒排列表
        for (int i = 0; i < wordsFreq.cols; i++){//i为列号，即第i个视觉中心
            int count = 0;
            for(int j = 0; j < wordsFreq.rows; j++){//j为行号，即第j张图片
                if(wordsFreq.at<float>(j, i) != 0 ){
                    //cout<<"不为零"<<endl;
                    invertedList[i].push_back(j);
                    count++;//临时变量，用来记录含有该聚类中心的图片数
                }
            }
           // idfMat.at<float>(0,i) = trainingImgNumber*1.0/count;
            idfMat.at<float>(0,i) = 170*1.0/count;
        }
        
        cout<<" < 构建倒排列表完成 > "<<endl;
        
        cout<<" < 倒排列表写入文件 > "<< endl;
        
        ofstream out;
        out.open("/Users/xuhuaiyu/Development/trainingResults/invertedList.txt");
        for(int i = 0 ; i < invertedList.size(); i ++){
            for(int j = 0; j < invertedList[i].size(); j++){
                out << invertedList[i][j] << " " ;
            }
            out << -1 << " ";//每行以-1作为标志位
        }
        out.close();
        cout<<"< 倒排列表写入文件完成 >"<< endl;
        
        //列优先遍历词频向量矩阵，计算每张图片的TF-IDF向量
        cout<<"< 计算每幅图片TF-IDF向量 >"<<endl;
        Mat tf_idf(wordsFreq.rows,wordsFreq.cols,CV_32F);//初始化tf_idf矩阵
        for(int i = 0 ; i < wordsFreq.cols; i++){//i为列号，即第i个视觉中心
            for(int j = 0 ; j < wordsFreq.rows; j++){//j为行号，即第j张图片
                tf_idf.at<float>( j, i ) = wordsFreq.at<float>(j,i) * idfMat.at<float>(0,i);
            }
        }
        cout<<"< TF-IDF向量构建完毕 >"<<endl;
        
        
        
        cout<<" < 聚类中心IDF数据写入文件 > "<<endl;
        fs.open("/Users/xuhuaiyu/Development/trainingResults/IDF.yml", FileStorage::WRITE);
        fs << "IDF" << idfMat;
        fs.release();
        cout<<" < 聚类中心IDF数据写入文件完毕 > "<<endl;
        
        cout<<"< 训练集图片TF-IDF向量写入文件。。。 >"<<endl;
        fs.open("/Users/xuhuaiyu/Development/trainingResults/TF-IDF.yml", FileStorage::WRITE);
        fs << "TF-IDF" << tf_idf;
        fs.release();
        cout << " <TF-IDF 写入文件完毕 > " << endl;
        
    }
    
    else if(option == '4'){
        string imgPath ;
        cout<<"请输入所需要匹配的图片路径："<<endl;
        cin >> imgPath;
        vector<int> idxs = matchImg(imgPath);
        

        ImgSet imgSet ;
        

        imgSet.loadImgsFromDir(trainingImgDir) ;//加载训练集中的所有图片
        vector<Mat> _imgs;
        for(int i = 0; i < idxs.size();i++){
            string imgName = imgSet.getImgName(idxs[i]);
            Mat image = imread(imgName);
            _imgs.push_back(image);
        }

        string _winName = "相似度匹配结果";


        imshowMany(_winName, _imgs);

        waitKey();
    }
    return 0;
}


/**
 提取所有图片的SIFT特征：
 
 dir:图片训练集的目录
 imgsDiscroptor: 所有图片的特征点结构体
 
 return：所有图片的特征点的特征描述符
 */

Mat extractSIFTDescriptor(string dir, ImgSet imgSet, list<struct imgInfo>& imgs){
    Mat allDescriptors  ;//将所有图片的特征描述符存储在一起
    
    cv::initModule_nonfree();//使用SIFT/SURF create之前，必须先initModule_<modulename>();

    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );//创建SIFT特征检测子（关键点）
    
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SIFT" );
    if( detector.empty() || descriptorExtractor.empty() )
    {
        cout << "Can not create detector or descriptor extractor of given types" << endl;
        return allDescriptors;//此处为提取特征点的特征描述符失败
    }
    
    
    

    
    cout<<"<开始提取SIFT特征>"<<endl;

    for(int i = 0 ; i < trainingImgNumber; i++){
        
        string imgName = imgSet.getImgName(i);
        cout<<"<第"<<i+1<<"张：图片名："<<imgName<<">"<<endl;
        Mat img = imread(imgName);
        
        struct imgInfo image ;//临时变量，用来存储当前图片的相关信息
        vector<KeyPoint> keypoint ;
        detector->detect( img, keypoint);//关键点
        cout<<"特征点个数:"<<keypoint.size()<<endl<<endl;
        image.keypoint = keypoint;
        
        Mat descriptors;//特征描述子
        descriptorExtractor->compute( img, keypoint, descriptors );
        image.descriptors = descriptors;
        imgs.push_back(image);

        allDescriptors.push_back(descriptors);//将该图像所有特征点的特征描述符附加到总特征描述符矩阵的末尾
        
    }
    
    //------------------------------------------------------------------------------------------
    
    return allDescriptors;
    
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


Mat h_kmeans(Mat& allDescriptors, int clusterNum){
    cvflann::KMeansIndexParams kmeans_param;
    Mat centers(clusterNumber,128,CV_32F);
    int true_number_clusters = flann::hierarchicalClustering<flann::L2<float>>(allDescriptors, centers, kmeans_param);//真正提取的特征值数量
    cout<<"共聚类成"<<true_number_clusters<<"类"<<endl;
    return centers.rowRange(cv::Range(0,true_number_clusters));
}


Mat countWordsFreq(Mat &img, vector<KeyPoint> keypoint, vector<vector<int>>* pointIdxsOfClusters, BOWImgDescriptorExtractor bowDE){
    Mat  imgDiscriptor;//该矩阵只有一行，即为normalized histogram,即TF
    
    bowDE.compute(img, keypoint, imgDiscriptor, pointIdxsOfClusters);
    return imgDiscriptor;
}

void buildWordsList(int wordsNum, struct visualWord* wordsList){
    wordsList = new struct visualWord[wordsNum];
    for(int i = 0 ; i < wordsNum ; i ++ ){//initialize the wordsList
        //        wordsList[i].imgPath = wordsFreq[i].imgPath;
        wordsList[i].next = NULL;
    }
}



/**
 倒排索引写入文件
 
 wordsList: 倒排索引
 */
#define invertedListFile "/Users/xuhuaiyu/Development/trainingResults/wordsFreq"
int invertedListIntoFile(struct visualWord* wordsList){
    /**
     1.写入一个int num，记录该单词对应的图片数
     2.写入num个struct invertedTableNode
     */
//    FILE* fp = fopen(invertedListFile, "a+");
//    int wordsNum = clusterNumber;
//    for(int i = 0; i < wordsNum; i ++ ){
//        
//    }
//    fclose(fp);
    
    ofstream out ;
    out.open(invertedListFile, ios::trunc | ios::out | ios::app);
    for(int i = 0 ; i < clusterNumber ; i ++ ){
        int num = wordsList[i].imgAmt;

        cout<< "num:"<<num<<endl;
        out << num;
        struct invertedTableNode * tmpNode = wordsList[i].next;
        while( tmpNode != NULL ){
            cout<<"tmpNode->picId "<<tmpNode->picId<<endl;
            out << tmpNode->picId;
            tmpNode = tmpNode->next;
        }
    }
    out.close();
    return 0;
}

typedef struct pic{
    int id;
    double similarity;
    
    bool operator <(const pic &other)const   //升序排序
    {
        return similarity<other.similarity;
    }
    bool operator >(const pic &other)const   //降序排序
    {
        return similarity>other.similarity;
    }
}pic;

vector<int> matchImg(string imgPath){
    Mat tfIdfOfBase;//训练集中每幅图片的tf_idf
    
    
    cout<<"读入训练集的tf_idf向量集"<<endl;
    FileStorage fs("/Users/xuhuaiyu/Development/trainingResults/TF-IDF.yml", FileStorage::READ);//读入训练集的tf_idf向量集
    fs["TF-IDF"] >> tfIdfOfBase;
    fs.release();
    cout<<"读取完毕 "<<endl;
    
    cout<<"tfIdfOfBase:"<<tfIdfOfBase.rows<<"*"<<tfIdfOfBase.cols<<endl;
    
    cout<<"读入视觉单词表"<<endl;
    Mat dictionary;
    fs.open("/Users/xuhuaiyu/Development/trainingResults/dictionary_test.yml", FileStorage::READ);//读入训练过程中构建的视觉单词表
    fs["vocabulary"] >> dictionary;
    fs.release();
    cout<<"读取完毕"<<endl;
    
    cout<<"计算测试图片的TF-IDF"<<endl;
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    Ptr<DescriptorMatcher>  matcher = DescriptorMatcher::create("BruteForce");
    BOWImgDescriptorExtractor bowDE(extractor, matcher);
    bowDE.setVocabulary(dictionary);
    
//    cout<<"imgPath"<<imgPath<<endl;
    Mat img = imread(imgPath);
    vector<KeyPoint> keypoint;
    detector->detect(img,keypoint);
    Mat TF = countWordsFreq(img, keypoint, NULL, bowDE);//获得传入的图片的TF
//    cout<<"TF.rows:"<<TF.rows<<"TF.cols"<<TF.cols<<endl;
    Mat idfMat ;//读入训练时候得到的各聚类中心的IDF
    fs.open("/Users/xuhuaiyu/Development/trainingResults/IDF.yml", FileStorage::READ);
    fs["IDF"] >> idfMat;
    fs.release();
//    cout<<"IDF.rows"<<idfMat.rows<<"IDF.ols"<<idfMat.cols<<endl;
    Mat tfIdfOfImg(1,clusterNumber,CV_32F);
//    cout<<"tf_idf.rows"<<tfIdfOfImg.rows<<"tf_idf.cols"<<tfIdfOfImg.cols<<endl;

    for(int i = 0 ; i < clusterNumber; i ++ ){//计算传入的图片的tf-idf向量
        tfIdfOfImg.at<float>(0,i) = idfMat.at<float>(0,i)*TF.at<float>(0,i);
    }
    cout<<"计算完毕"<<endl;
    //读入倒排索引------------------------------------------------------------------------------------------
    cout<<"读取倒排索引"<<endl;
    vector<vector<int>> invertedList(clusterNumber);
    ifstream in;
    int value;
    in.open("/Users/xuhuaiyu/Development/trainingResults/invertedList.txt");
    for( int i = 0 ; i < invertedList.size(); i ++ ){
        in >>value;
        while(value != -1){
            
            invertedList[i].push_back(value);
            in >> value ;
        }
    }
    in.close();
    cout<<"读取完毕"<<endl;

    cout<<"开始相似图片匹配"<<endl;

    vector<pic> results(6);

    int flag[clusterNumber] = {0};
    for(int i = 0 ; i < tfIdfOfImg.cols; i ++ ){
        
        if(tfIdfOfImg.at<float>(0,i) != 0){//若该图片内包含有第i个视觉单词
            vector<int> imgsId = invertedList[i];//获得倒排索引中，第i个聚类中心对应的所有图片的id
            for(int j = 0 ; j < imgsId.size(); j ++ ){//遍历倒排索引中包含该视觉单词的所有图片
                int id = imgsId[j];
                if(flag[j] == 0){
                    flag[j] = 1;
                    double vectorMultiply = 0.00;//词频向量向量积
                    double vector1Modulo = 0.00;//待检测图片词频向量向量积
                    double vector2Modulo = 0.00;//图库内图片词频向量向量积
                    
                    for( int t = 0 ; t < clusterNumber; t++){//计算余弦相似度
                        
                        vector1Modulo += 0.1 * tfIdfOfBase.ptr<float>(id)[t] * tfIdfOfBase.ptr<float>(id)[t];
                        vector2Modulo += 0.1 * tfIdfOfImg.at<float>(0,t) * tfIdfOfImg.at<float>(0,t);
                        vectorMultiply += 0.1 * tfIdfOfBase.ptr<float>(id)[t] * tfIdfOfImg.at<float>(0,t);
                        
                    }
                    
                    vector1Modulo = sqrt(vector1Modulo);
                    vector2Modulo = sqrt(vector2Modulo);
                    double temp = vectorMultiply/(vector1Modulo*vector2Modulo);
                    for(int t = 5; t >=0; t--){
                        if(results[t].similarity < temp){
                            results[0].id = id;
                            results[0].similarity = temp;
                            break;
                        }
                    }
                    sort(results.begin(),results.end());
                 }
            }
            
        }
    }
    vector<int> ids;
    cout<<"from matImg"<<endl;
    
    for(int i = 5 ; i >= 0 ; i-- ){
        cout<<results[i].id<<" ";
        ids.push_back(results[i].id);
    }
    return ids;
//    return k;
}
int cmp(const pair<int,float>& x, const pair<int,float>& y){
    return x.second < y.second;
}
void imshowMany(const std::string& _winName, const vector<Mat>& _imgs)
{
    int nImg = (int)_imgs.size();
    
    Mat dispImg;
    int size;
    int x, y;
    // w - Maximum number of images in a row
    // h - Maximum number of images in a column
    int w, h;
    // scale - How much we have to resize the image
    float scale;
    int max;
    if (nImg <= 0)
    {
        printf("Number of arguments too small....\n");
        return;
    }
    else if (nImg > 12)
    {
        printf("Number of arguments too large....\n");
        return;
    }
    
    else if (nImg == 1)
    {
        w = h = 1;
        size = 300;
    }
    else if (nImg == 2)
    {
        w = 2; h = 1;
        size = 300;
    }
    else if (nImg == 3 || nImg == 4)
    {
        w = 2; h = 2;
        size = 300;
    }
    else if (nImg == 5 || nImg == 6)
    {
        w = 3; h = 2;
        size = 300;
    }
    else if (nImg == 7 || nImg == 8)
    {
        w = 4; h = 2;
        size = 200;
    }
    else
    {
        w = 4; h = 3;
        size = 150;
    }
    dispImg.create(Size(100 + size*w, 60 + size*h), CV_8UC3);
    for (int i= 0, m=20, n=20; i<nImg; i++, m+=(20+size))
    {
        x = _imgs[i].cols;
        y = _imgs[i].rows;
        max = (x > y)? x: y;
        scale = (float) ( (float) max / size );
        if (i%w==0 && m!=20)
        {
            m = 20;
            n += 20+size;
        }
        Mat imgROI = dispImg(Rect(m, n, (int)(x/scale), (int)(y/scale)));
        resize(_imgs[i], imgROI, Size((int)(x/scale), (int)(y/scale)));
    }
    namedWindow(_winName);
    imshow(_winName, dispImg);
}


/**
 计算余弦相似度
 
 srcFreq:待检测图片的词频向量
 desFreq：图片库中图片的词频向量
 
 return:余弦相似度
 */
double cosSimlarity(int* srcFreq, int* dstFreq, int wordsNum){
    double vectorMultiply = 0.00;//词频向量向量积
    double vector1Modulo = 0.00;//待检测图片词频向量向量积
    double vector2Modulo = 0.00;//图库内图片词频向量向量积
    
    for(int i = 0 ;i < wordsNum; i ++ ){
        vector1Modulo += 0.1 * srcFreq[i] * srcFreq[i];
        vector2Modulo += 0.1 * dstFreq[i] * dstFreq[i];
        
        vectorMultiply += 0.1 * srcFreq[i] * dstFreq[i];
    }
    vector1Modulo = sqrt(vector1Modulo);
    vector2Modulo = sqrt(vector2Modulo);
    
    return vectorMultiply/(vector1Modulo*vector2Modulo);
}


/**
 总结1：
 在使用flann::hierarchicalClustering<Distance>进行分层聚类时候
 三个参数的方法已经被弃用（deprecated），源代码中将返回值删除，若使用会出现无返回值的错误
 int hierarchicalClustering(const Mat& features, Mat& centers, const ::cvflann::IndexParams& params)
 
 所以可使用的为四个参数的方法：
 int hierarchicalClustering(const Mat& features, Mat& centers, const ::cvflann::IndexParams& params,
 Distance d = Distance())
 
 在调用的时候，flann::hierarchicalClustering<float,float>(features, centers, kmeans_param);的调用方式已经不可用
 需要使用flann::hierarchicalClustering<flann::L2<float>>(features, centers, kmeans_param);的调用方式
 */










