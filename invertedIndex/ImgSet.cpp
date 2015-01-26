//
//  ImgSet.cpp
//  invertedIndex
//
//  Created by xuhuaiyu on 14-12-4.
//  Copyright (c) 2014年 xuhuaiyu. All rights reserved.
//

#include "ImgSet.h"
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <assert.h>

void ImgSet:: loadImgsFromDir(string direcroty){
    const char* dirname = direcroty.c_str();
//    cout<<"<训练集目录为"<< dirname<< ">"<<endl;
    assert(dirname != NULL);
    
    char path[512];
    struct dirent *filename;//readdir 的返回类型
    DIR *dir;//血的教训阿，不要随便把变量就设成全局变量。。。。
    
    dir = opendir(dirname);
    if(dir == NULL)
    {
        printf("open dir %s error!\n",dirname);
        exit(1);
    }
    
    while((filename = readdir(dir)) != NULL)
    {
        //跳过这两个目录
        if(!strcmp(filename->d_name,".")||!strcmp(filename->d_name,".."))
            continue;
        
        //非常好用的一个函数，比什么字符串拼接什么的来的快的多
        sprintf(path,"%s/%s",dirname,filename->d_name);
        
        struct stat s;
        lstat(path,&s);
        
        if(S_ISDIR(s.st_mode))
        {
            loadImgsFromDir(path);//递归调用
        }
        else
        {
//            cout<<path<<endl;
            ++imgNum;
            imgs.push_back(string(path));
        }
    }
    closedir(dir);

}