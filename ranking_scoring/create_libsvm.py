import numpy as np
import pandas as pd
import os
import time

'''
创建训练数据集
qid对应pairs数据中表的行索引值
libsvm数据格式如下：
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
'''

# 对比序列信息文件路径hah
# INPUT_PAIR_DATAS = "../data_process/data/placepulse/out/wealthy_votes_filter.csv"
INPUT_PAIR_DATAS = "votes/bd_votes_fuse.csv"
# 图片瓶颈层信息文件路径
BOTTLENECK_DIR = "../model_process/bottleneck/inception_resnet_baidu_dir/btlk1001"
# libsvm文件输出地址
OUTPUT_PAIR_DIR = "./libsvm/inception_resnet_baidu_dir/btlk1001/"
OUTPUT_PAIR_DATAS = OUTPUT_PAIR_DIR + "wealthy"


# 根据图片名称和模型名称，获取对应路径下的瓶颈向量值
def get_segmentation_values(img_name):
    bottleneck_path = os.path.join(BOTTLENECK_DIR, img_name) + '.jpg.txt'
    # print(bottleneck_path)
    # 直接从文件中获取图片相应的特征向量。
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    return bottleneck_string


def get_bottleneck_values(img_name):
    bottleneck_path = os.path.join(BOTTLENECK_DIR, img_name) + '.jpg.txt'
    # print(bottleneck_path)
    # 直接从文件中获取图片相应的特征向量。
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


# 语义分割模型得到的特征向量
def save_pair_segmentation_values(rst, index, valuesA, valuesB, data_file, is_training):
    rankA = 1
    rankB = 1
    # print(rst)
    if rst == 'left':
        rankA = 2
        rankB = 0
    elif rst == 'right':
        rankA = 0
        rankB = 2
    else:
        if not is_training:
            return
    bottleneckA_str = str(rankA) + " qid:" + str(index + 1) + " " + valuesA
    bottleneckB_str = str(rankB) + " qid:" + str(index + 1) + " " + valuesB

    with open(data_file, 'a') as writer:
        writer.write(bottleneckA_str + '\n')
        writer.write(bottleneckB_str + '\n')


# is_training:true存储相等的对比序列, false 移除相等和unknown对比序列
def save_pair_bottleneck_values(rst, index, valuesA, valuesB, data_file, is_training):
    rankA = 1
    rankB = 1
    # print(rst)
    if rst == 'left':
        rankA = 2
        rankB = 0
    elif rst == 'right':
        rankA = 0
        rankB = 2
    else:
        if not is_training:
            return
    bottleneckA_str = str(rankA) + " qid:" + str(index + 1)
    bottleneckB_str = str(rankB) + " qid:" + str(index + 1)

    for i in range(len(valuesA)):
        if valuesA[i] != 0:
            bottleneckA_str += " " + str(i + 1) + ":" + str(valuesA[i])
        if valuesB[i] != 0:
            bottleneckB_str += " " + str(i + 1) + ":" + str(valuesB[i])
    with open(data_file, 'a') as writer:
        writer.write(bottleneckA_str + '\n')
        writer.write(bottleneckB_str + '\n')


# 创建指定区间的数据集
def create_libsvm(first_index, last_index, is_training):
    readData = pd.read_csv(INPUT_PAIR_DATAS, header=None)
    if last_index > 0:
        mat_datas = readData.values[first_index:last_index, :]
        length = last_index - first_index
    else:
        mat_datas = readData.values[:, :]
        length = len(mat_datas[:])
        first_index = 0
        last_index = length
    data_file = OUTPUT_PAIR_DATAS + str(first_index) + "_" + str(last_index) + ".txt"
    print(data_file)
    time1 = time.time()
    time0 = time1
    for i in range(length):
        try:
            # if not os.path.exists(data_file):
            #     os.mkdir(data_file)
            # bottleneck_valuesA = get_segmentation_values(mat_datas[i][0])
            # bottleneck_valuesB = get_segmentation_values(mat_datas[i][1])
            # save_pair_segmentation_values(mat_datas[i][2], i, bottleneck_valuesA, bottleneck_valuesB, data_file,
            #                               is_training)
            bottleneck_valuesA = get_bottleneck_values(mat_datas[i][0])
            bottleneck_valuesB = get_bottleneck_values(mat_datas[i][1])
            save_pair_bottleneck_values(mat_datas[i][2], i, bottleneck_valuesA, bottleneck_valuesB, data_file,
                                        is_training)
            if i % 10000 == 0:
                print("time: %s" % (time.time() - time1))
                time1 = time.time()
        except:
            print("error: %s  and  %s" % (mat_datas[i][0], mat_datas[i][1]))
    time2 = time.time()
    print("total time: ", time2 - time0)


# 构建一组对比数据
def create_one_libsvm(array_data, cateragy, is_training, i):
    data_file = OUTPUT_PAIR_DATAS + cateragy + ".txt"
    try:
        bottleneck_valuesA = get_segmentation_values(array_data[0])
        bottleneck_valuesB = get_segmentation_values(array_data[1])
        # bottleneck_valuesA = get_bottleneck_values(array_data[0])
        # bottleneck_valuesB = get_bottleneck_values(array_data[1])
        # if not os.path.exists(data_file):
        #     os.mkdir(data_file)
        save_pair_segmentation_values(array_data[2], i, bottleneck_valuesA, bottleneck_valuesB, data_file, is_training)
        # save_pair_bottleneck_values(array_data[2], i, bottleneck_valuesA, bottleneck_valuesB, data_file, is_training)
    except:
        print("error: %s  and  %s" % (array_data[0], array_data[1]))


# 随机化标注数据集
def random_votes_dataset(out_path):
    test_percent = 20
    vali_percent = 10
    readData = pd.read_csv(INPUT_PAIR_DATAS, header=None)
    mat_datas = readData.values[:, :]
    data_len = len(mat_datas[:])
    for i in range(data_len):
        chance = np.random.randint(100)
        cateragy = "none"
        if chance < test_percent:
            cateragy = "f_test"
        elif chance < (test_percent + vali_percent):
            cateragy = "f_vali"
        else:
            cateragy = "f_train"
        if mat_datas[i][2] != 'equal':
            data_file = out_path + cateragy + ".csv"
            with open(data_file, 'a') as writer:
                writer.write(mat_datas[i][0] + "," + mat_datas[i][1] + "," + mat_datas[i][2] + '\n')


# 创建随机的数据集
def create_random_dataset():
    test_percent = 30
    vali_percent = 10
    readData = pd.read_csv(INPUT_PAIR_DATAS, header=None)
    mat_datas = readData.values[:, :]
    data_len = len(mat_datas[:])
    print("all data : %s" % data_len)
    time1 = time.time()
    for i in range(data_len):
        chance = np.random.randint(100)
        if chance < test_percent:
            create_one_libsvm(mat_datas[i], '1_test', False, i)
        elif chance < (test_percent + vali_percent):
            create_one_libsvm(mat_datas[i], '1_vali', False, i)
        else:
            create_one_libsvm(mat_datas[i], '1_train', False, i)
        if i % 10000 == 0:
            print("step %s time %s " % (i, time.time() - time1))
            time1 = time.time()


# 将百度数据与谷歌数据混合在一起， num为数据条数
def create_fuse_libsvm(num, category):
    bd_lines = open("bottleneck/bd_mobilenet_v1/wealthy_allvotespairs_libsvm_%s.txt" % category, "r").readlines()
    gg_lines = open("bottleneck/mobilenet_v1/wealthy_libsvm_%s.txt" % category, "r").readlines()
    bd_len = len(bd_lines)
    gg_len = len(gg_lines)
    print("%s   ---   %s" % (bd_len, gg_len))
    bd = 0
    with open("bottleneck/bd_mobilenet_v1/wealthy_fuse_%s.txt" % category, "a") as new_file:
        for i in range(num):
            if i % 2 == 0:
                if bd >= bd_len:
                    continue
                new_file.write(bd_lines[bd])
                new_file.write(bd_lines[bd + 1])
                bd = bd + 2
            else:
                new_file.write(gg_lines[gg_len - 1])
                new_file.write(gg_lines[gg_len - 2])
                gg_len = gg_len - 2


def set_by_name(imgs_dir, last):
    """获取文件夹下所有文件名集合"""
    set_download = set()
    img_list = list()
    for root, dirs, files in os.walk(imgs_dir):
        for file in files:
            name = str(file)[:-last]
            set_download.add(name)
            img_list.append(name)
    print("total images num：%s" % len(set_download))  # 打印各个文件夹图片量
    return set_download, img_list


def filter_libsvm():
    """移除未得到特征文件的标注数据"""
    readData = pd.read_csv(INPUT_PAIR_DATAS, header=None)
    mat_datas = readData.values[:, :]
    data_len = len(mat_datas[:])
    bottleneck_set, _ = set_by_name(BOTTLENECK_DIR, 8)
    print(len(bottleneck_set))
    out_csv = "./libsvm/google_wealthy_dataset/wealthy_db_test_filter.csv"
    for i in range(data_len):
        data = mat_datas[i]
        if data[0] in bottleneck_set and data[1] in bottleneck_set:
            with open(out_csv, 'a') as writer:
                out_string = ','.join(str(x) for x in data)
                writer.write(out_string + '\n')
        else:
            print('%s, %s' % (data[0], data[1]))


# 注意数据是追加写入文件，重新生成时需删除已有数据
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PAIR_DIR):
        os.mkdir(OUTPUT_PAIR_DIR)
    # outpath = "./votes/placepulse/monilenetv2_wealthy_votes"
    # random_votes_dataset(outpath)

    # 9534  2627  1288
    # 7525  2174  1034

    # INPUT_PAIR_DATAS = "./votes/bd_votes_vali.csv"
    # create_libsvm(0, 1288, False)
    # INPUT_PAIR_DATAS = "./votes/bd_votes_test.csv"
    # create_libsvm(0, 2627, False)
    # INPUT_PAIR_DATAS = "./votes/bd_votes_train.csv"
    # create_libsvm(0, 9534, False)

    """谷歌固定数据集构建"""
    """
    INPUT_PAIR_DATAS = "./libsvm/google_wealthy_dataset/wealthy_db_train_filter.csv"
    create_libsvm(0, 0, False)
    INPUT_PAIR_DATAS = "./libsvm/google_wealthy_dataset/wealthy_db_vali_filter.csv"
    create_libsvm(0, 0, False)
    INPUT_PAIR_DATAS = "./libsvm/google_wealthy_dataset/wealthy_db_test_filter.csv"
    create_libsvm(0, 0, False)
    """

    """百度固定数据集构建"""
    """
    INPUT_PAIR_DATAS = "./libsvm/baidu_wealthy_dataset/wealthy_db_train.csv"
    create_libsvm(0, 0, False)
    INPUT_PAIR_DATAS = "./libsvm/baidu_wealthy_dataset/wealthy_db_vali.csv"
    create_libsvm(0, 0, False)
    INPUT_PAIR_DATAS = "./libsvm/baidu_wealthy_dataset/wealthy_db_test.csv"
    create_libsvm(0, 0, False)
    """
    # create_libsvm(0, 0, False)

    # filter_libsvm()

    INPUT_PAIR_DATAS = "./predict/bd_votes_pairs.csv"
    create_libsvm(0, 0, True)

    # create_random_dataset()

    # create all predict libsvm

    # create_libsvm(0, 50000, False)
    # create_libsvm(50000, 280000, False)
    # create_libsvm(280000, 364000, False)

    # create_libsvm(90000, 120000, True)
    # create_libsvm(120000, 150000, True)
    # create_libsvm(150000, 177618, True)

    # create_fuse_libsvm(2500, 'vali')
    # create_fuse_libsvm(5000, 'test')
    # create_fuse_libsvm(20000, 'train')

    # print("create train dataset !")
    # create_libsvm(0, 28000, True)
    # # print("create vali dataset !")
    # create_libsvm(28000, 32000, False)
    # # print("create test dataset !")
    # create_libsvm(32000, 40000, False)

    # create_libsvm(40000, 130000, False)
