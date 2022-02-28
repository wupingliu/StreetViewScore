#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : IDEA
#   Author      : LWP
#   Created date: 2019-9-27 16:08
#   Description : 通过已知的baidu坐标表格，爬取百度街景图片。
#================================================================

import urllib, requests
import numpy as np
import pandas as pd
import threading
import os
import data_process.util.transform as transform


class BaiduDownload:
    def __init__(self, inpath, outpath, sdpath, flpath):
        self.APIMETA = 'https://mapsv0.bdimg.com/?'
        self.APIIMG = "https://mapsv0.bdimg.com/?"
        self.THREAD_NUM = 50  # 线程数
        self.BATCH_NUM = 100
        self.HEAD_NUM = 8  # 图片朝向，默认8个
        self.IMG_width = 480  # 单张图片宽
        self.IMG_height = 360  # 单张图片高
        self.in_path = inpath  # 输入去重后的坐标点文件
        self.sd_path = sdpath  # 输出爬取成功坐标点文件, 临时存储
        self.fl_path = flpath  # 输出爬取失败坐标点文件，临时存储
        self.out_path = outpath  # 输出图片的路径
        self.list_succeed = list()
        self.list_failed = list()

    # 下载一张图片
    def download_one_img(self, img_info, index, thread_id):
        requests.adapters.DEFAULT_RETRIES = 50
        sess = requests.session()
        sess.keep_alive = False
        img_mc = transform.transformM2BM(img_info)  # 火星坐标转为百度墨卡托坐标
        params_road = dict(qt='qsdata', x='{}'.format(img_mc[0]), y='{}'.format(img_mc[1]))
        url_road = self.APIMETA + urllib.parse.urlencode(params_road)
        try:
            response_road = requests.get(url_road, stream=True, verify=True).json()  # json文件包
            status_road = response_road['result']['error']
            if status_road == 0:
                pid = response_road['content']['id']
                point_x = response_road['content']['x']
                point_y = response_road['content']['y']
                for i in range(self.HEAD_NUM):
                    # 根据每个点下载图片数来确定朝向
                    heading = 360 / self.HEAD_NUM * i
                    params_img = dict(qt='pr3d', fovy=35, quality=70, panoid='{}'.format(pid), heading=heading, pitch=0,
                                      width=self.IMG_width, height=self.IMG_height)
                    url_img = self.APIIMG + urllib.parse.urlencode(params_img)
                    response_img = requests.get(url_img, verify=True)  # html网页
                    status_code = response_img.status_code
                    if status_code == 404:
                        print("thread_id %s ***** %s_%s_%s.jpg  index: %s*********request failed!" % (
                        thread_id, img_info[0], img_info[1], heading, index))
                    else:
                        # 将图片存储到对应位置
                        # print("thread_id %s ***** %s_%s_%s.jpg  index: %s*********request succeed!" % (thread_id, img_info[0], img_info[1], heading, index))
                        with open(self.out_path + "/%.11s_%.10s_%s_%s.jpg" % (point_x, point_y, int(heading), index),
                                  'wb') as img_file:
                            img_file.write(response_img.content)
                # 下载成功存储：图片ID、原经度、原纬度、点在原文件编号、百度X坐标、百度Y坐标
                point_info = [index, img_info[0], img_info[1], pid, point_x, point_y]
                self.list_succeed.append(point_info)
                if len(self.list_succeed) % 100 == 0:
                    np.savetxt(self.sd_path, np.mat(self.list_succeed), fmt="%s", delimiter=",")
            else:
                print("thread_id %s *******11111**********request failed!***********%s" % (thread_id, index))
                # 下载失败存储：失败标记、原经度、原纬度、点在原文件编号
                point_fail = [index, img_info[0], img_info[1], 111]
                self.list_failed.append(point_fail)
                if len(self.list_failed) % 100 == 0:
                    np.savetxt(self.fl_path, np.mat(self.list_failed), fmt="%s", delimiter=",")
            return 2
        except:
            print("thread_id %s *******22222**********request failed!***********%s" % (thread_id, index))
            # 下载失败存储：失败标记、原经度、原纬度、点在原文件编号
            point_fail = [index, img_info[0], img_info[1], 222]
            self.list_failed.append(point_fail)
            if len(self.list_failed) % 100 == 0:
                np.savetxt(self.fl_path, np.mat(self.list_failed), fmt="%s", delimiter=",")
            return 0

    # 下载一个batch的图片，由点所在的“start-->end”索引决定
    def download_batch_img(self, all_img_info, start, end, thread_id):
        length = len(all_img_info[:])
        for index in range(length)[start:end]:
            img_temp = all_img_info[index]
            img_info = [img_temp[1], img_temp[2]]
            self.download_one_img(img_info, index, thread_id)

    # 下载所有坐标点文件对应的街景
    def download_all_img(self, f_path):
        threads = []
        read_data = pd.read_csv(f_path, header=None, encoding='gbk')
        length = len(read_data)
        mat_datas = read_data.values[:, :]
        num = round(length / self.THREAD_NUM)
        for i in range(self.THREAD_NUM):
            start_num = i * num
            if i == self.THREAD_NUM - 1:
                end_num = length
            else:
                end_num = (i + 1) * num
            t = threading.Thread(target=self.download_batch_img, args=(mat_datas, start_num, end_num, i))
            threads.append(t)
            # img_dir = ('imgs/thread%s/' % i)
            # mkdir(img_dir)
        for t in threads:
            t.setDaemon(True)
            t.start()
        # 线程结束后关闭
        for t in threads:
            t.join()


class DataProcess:
    def __init__(self, input_path, filter_path, succeed_path, failed_path):
        self.input_path = input_path
        self.filter_path = filter_path
        self.succeed_path = succeed_path
        self.failed_path = failed_path

    # 过滤掉位置重复的坐标点
    def filter_points(self):
        read_datas = pd.read_csv(self.input_path, header=None, encoding='gbk')
        length = len(read_datas)
        mat_datas = read_datas.values[:, :]
        filter_set_point = set()
        filter_list_point = list()
        for l in range(length):
            point = ('%.9s_%.8s' % (mat_datas[l, 1], mat_datas[l, 2]))
            if point not in filter_set_point:
                filter_set_point.add(point)
                filter_list_point.append(mat_datas[l])
        mat_filter_point = np.mat(filter_list_point)
        np.savetxt(self.filter_path, mat_filter_point, fmt="%s", delimiter=",", encoding='gbk')

    # 获取文件中所有坐标点信息
    def get_all_points(self, file_path):
        if not os.path.exists(file_path):
            return None
        all_points = pd.read_csv(file_path, encoding='gbk')
        mat_points = all_points.values[:, :]
        set_all_points = set()
        length = len(all_points)
        for i in range(length):
            set_all_points.add('%.9s_%.8s' % (mat_points[i, 1], mat_points[i, 2]))
        return set_all_points

    # 遍历查询得到未下载街景的坐标点信息;  另一种方式：直接读取未下载的坐标点文件也可以
    def search_not_download(self):
        all_succeed = self.get_all_points(self.succeed_path)
        read_datas = pd.read_csv(self.filter_path, header=None, encoding='gbk')
        length = len(read_datas)
        list_not_download = []
        if all_succeed is None:
            all_succeed = set()
        mat_datas = read_datas.values[:, :]
        for l in range(length):
            name = ('%.9s_%.8s' % (mat_datas[l, 1], mat_datas[l, 2]))
            if name not in all_succeed:
                list_not_download.append([l, mat_datas[l, 1], mat_datas[l, 2], 333])
        mat_not_download = np.mat(list_not_download)
        np.savetxt(self.failed_path, mat_not_download, fmt="%s", delimiter=",")
        if len(list_not_download) > 4000:
            return 2
        else:
            return 1


if __name__ == '__main__':
    in_path = "csv/200WGS_Points.csv"  # 原始坐标点文件输入路径
    ft_path = "csv/200WGS_Filter_Points.csv"  # 过滤完重复坐标点后文件路径
    sd_path = "csv/200WGS_Succeed_Points.csv"  # 下载成功坐标信息
    fl_path = "csv/200WGS_Failed_Points.csv"  # 下载失败坐标信息
    img_dir = "IMGS"  # 图片输出文件夹
    # 初始化数据处理类
    data_center = DataProcess(in_path, ft_path, sd_path, fl_path)
    # 过滤掉原始文件中重复的点，避免重复请求下载
    data_center.filter_points()
    # 初始化百度街景爬取类
    baidu_center = BaiduDownload(ft_path, img_dir, sd_path, fl_path)
    # 可重复多次下载请求，避免因网络等原因导致的下载停止
    for step in range(1):
        # 搜索未下载成功的数据
        search_code = data_center.search_not_download()
        if search_code == 2:
            # 清空失败信息存储器；新下载成功的则在原基础上增加
            baidu_center.list_failed.clear()
            # 多线程爬取图片
            baidu_center.download_all_img(fl_path)
            # 存储最终成功点信息
            succeed_mat = np.mat(baidu_center.list_succeed)
            np.savetxt(sd_path, succeed_mat, fmt="%s", delimiter=",")
            # 存储最终失败点信息
            failed_mat = np.mat(baidu_center.list_failed)
            np.savetxt(fl_path, failed_mat, fmt="%s", delimiter=",")
        else:
            break
