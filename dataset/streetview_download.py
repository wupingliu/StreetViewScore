#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : IDEA
#   Author      : LWP
#   Created date: 2019-9-27 16:08
#   Description : 对PlacePuls项目的街景数据进行下载，通过google街景的图片ID下载指定街景图片，
#                 图片ID由下载的标注数据表提供。
#================================================================

import pandas as pd
import numpy as np
import urllib,requests
import time
import threading
import os

# 访问google 街景需要的头文件及配置信息
APIPath = 'api.txt'  # 存储google下载所需的key值，网上下载的：www.
apiList = list(pd.read_csv(APIPath,header=None)[0])
#apiIndex = 1
APIMETA = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
APIIMG = "https://maps.googleapis.com/maps/api/streetview?"
proxy = {
    'http': 'http://127.0.0.1:19180',
    'https': 'http://127.0.0.1:19180'
}
#headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'} # This is chrome, you can set whatever browser you like
headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}

set_dowload_failed = set()

# 下载一张街景图片
# 参数：img_info——街景信息表中图片基本信息
#       index——下载图片在表中索引号
#       category——下载图片所属的类别
#       apiIndex——使用的google街景下载的API编号索引
#       threadID——该方法被指定ID的线程调用
def download_one_img(img_info,index,category,apiIndex,threadID):
    #img_info = './pp2_20161010/votes.csv'
    PARAMS = dict(size = "400x400", location = '{},{}'.format(img_info[1],img_info[2]), key = apiList[apiIndex])
    url_meta = APIMETA + urllib.parse.urlencode(PARAMS)
    url_img = APIIMG + urllib.parse.urlencode(PARAMS)
    #    print(url_img)
    requests.adapters.DEFAULT_RETRIES = 50
    s = requests.session()
    s.keep_alive = False
    response = None
    try:
        response = requests.get(url_meta,stream=True,verify=True,proxies=proxy,headers=headers).json()
        status = response['status']
        html=requests.get(url_img,verify=True,proxies=proxy,headers=headers)  #dowmload by requests
        res = html.status_code
    except:
        set_dowload_failed.add(img_info[0])
        res = 999
        print("threadID %s *****************request failed!******************** %s"%(threadID,index))
    if response == None:
        return 2
    if response['status'] == u'OK':
        if res == 200:
            # 街景图片保存到指定文件夹：./imgdatas/[category]_new_img下面
            with open("../dataCenter/google/placepulse/%s.jpg"%img_info[0], 'wb') as file:
                file.write(html.content)
            print("threadID %s *** %s_img/%s.jpg  num: %s"%(threadID,category,img_info[0],index))
            return 1
        elif res == 999:
            pass
            return 1
        else:
            set_dowload_failed.add(img_info[0])
            print("threadID %s ***** status_code : %s*******dowmload failed!***************** %s"%(threadID,res,index))
            return 2
    elif status == u'OVER_QUERY_LIMIT':
        print(res)
        set_dowload_failed.add(img_info[0])
        return 2
    elif status == u'ZERO_RESULTS':
        print(status)
        set_dowload_failed.add(img_info[0])
        return 0
    else:
        print("else")
        set_dowload_failed.add(img_info[0])
        return 0

# 指定一个线程需下载的街景图片总数
# 参数：all_img_info——街景信息表所有图片信息
#       category——下载图片所属的类别
#       start——信息表中图片起始索引
#       end——信息表中图片结束索引
#       threadID——该方法被指定ID的线程调用
def download_patch_img(all_img_info,category,start,end,threadID):
    apiIndex = 10
    length = len(all_img_info[:])
    #    print("%s; %s; %s; %s"%(length,start,end,threadID))
    for index in range(length)[start:end]:
        code = download_one_img(all_img_info[index],index,category,apiIndex,threadID)
        if code == 2:
            apiIndex = apiIndex + 1
            if apiIndex >= 15:
                apiIndex = apiIndex % 15

# 下载信息表中指定类别的所有街景图片
# 参数：category——下载图片所属的类别
#       info_path——图片信息表的本地路径
def download_all_img(category,info_path):
    threads = []
    readData = pd.read_csv(info_path)
    length = len(readData)
    mat_datas = readData.values[:,:]
    num = round(length/THREAD_NUM)
    for i in range(THREAD_NUM):
        start = i*num
        if i==(THREAD_NUM-1):
            end = length
        else:
            end = (i+1)*num
        #all_img_info = mat_datas[start:end,:]
        t = threading.Thread(target=download_patch_img,args=(mat_datas,category,start,end,i))
        threads.append(t)
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()


# 查询未成功下载的图片信息
# 参数：category——下载图片所属的类别
#       dataPath——需要比对的目标文件路径
def search_not_download(category,dataPath):
    set_download = file_name("../../dataCenter/google/placepulse/")
    list_not_download = []
    list_download = []
    readData = pd.read_csv(dataPath)
    length = len(readData)
    print(length)
    print(len(set_download))
    mat_datas = readData.values[:,:]
    for i in range(length):
        if mat_datas[i][0] not in set_download:
            #            print(mat_datas[i][0])
            list_not_download.append(mat_datas[i])
        elif mat_datas[i][1] not in set_download:
            list_not_download.append(mat_datas[i])
        else:
            list_download.append(mat_datas[i])
    print("the number of not download for %s : %s"%(category,len(list_not_download)))
    # mat_csv = np.mat(list_not_download)
    # np.savetxt("../data/placepulse/%s_not_img.csv"%category,mat_csv,fmt="%s",delimiter=",")
    # mat_csv = np.mat(list_download)
    # np.savetxt("../data/placepulse/%s.csv"%category,mat_csv,fmt="%s",delimiter=",")
    if len(list_not_download) > 2000:
        print("222222222")
        return 2
    else:
        print("111111111")
        return 1

# 遍历文件夹下所有.jpg图片名字，存储到集合保存
def file_name(file_dir):
    set_download = set()
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                set_download.add(os.path.splitext(file)[0])
    return set_download


# 从原始信息表中获取某一类图片所有信息存储到新的表格
# 参数：category——下载图片所属的类别
def get_votes_info(category):
    dataPath = '../data/placepulse/votes.csv'
    requestData = pd.read_csv(dataPath)
    length = len(requestData)
    mat_datas = requestData.values[:,:]
    list_all = []
    len_votes = 0
    for i in range(length):
        if mat_datas[i][7]==category:
            len_votes = len_votes + 1
            list_all.append(list(mat_datas[i]))
    print("number of %s votes: %s"%(category,len_votes))
    mat_csv = np.mat(list_all)
    out_path = "../data/placepulse/%s_info.csv"%category
    np.savetxt(out_path,mat_csv,fmt="%s",delimiter=",")
    return out_path


# 初始图片下载，重复请求多次，避免不可预估原因导致的线程结束下载
# 参数：category——下载图片所属的类别
#       dataPath——需要比对的目标文件路径
#       steps——需要轮询的次数，int
def download_main(category,dataPath,steps):
    download_all_img(category,dataPath)  # 先下载一次
    # 遍历未下载图片，进行多次重复请求
    for i in range(steps):
        search_code = search_not_download(category,dataPath)
        if search_code == 2:
            dataPath_not = ("../data/placepulse/all_not_info.csv")
            set_dowload_failed.clear()
            download_all_img(category,dataPath_not)
            failed_list = list(set_dowload_failed)
            mat_csv = np.mat(failed_list)
            np.savetxt("../data/placepulse/all_failed_info.csv", mat_csv,fmt="%s",delimiter=",")
        else:
            break


THREAD_NUM = 50     # 开启的线程数，根据计算机性能决定
BACK_STEPS = 100    # 反复对失败图片进行下载的最大次数
THRESHOLD = 5000   # 下载失败的图片数不超过域值时，认为爬取完成

 # depressing; wealthy; safety; lively; beautiful; boring
if __name__ == '__main__':
    start = time.time()
    category = u"wealthy"
    dataPath = "../data/placepulse/%s_info.csv" % category

    search_not_download(category, dataPath)

    # dataPath = get_votes_info(category)

    # download_main(category, dataPath, BACK_STEPS)
    # print("all over %.5f seconds" %(time.time()-start))
