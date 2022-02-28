import os
import glob
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import time

class BottleneckCenter:
    def __init__(self, in_dir, out_dir, input_size, is_preprocessed, img_input=None, img_tensor=None):
        self.input_dir = in_dir        # 原始图片输入目录
        self.output_dir = out_dir      # 特征向量输出路径
        self.INPUT_SIZE = input_size    # 一纬向量[n, w, h, l]
        self.IS_PROCESSED = is_preprocessed
        self.img_input = img_input
        self.img_process_tensor = img_tensor
        self.out_bottleneck_path = []
        self.image_data = []
        self.size_temp = self.INPUT_SIZE[0]
        pass

    # 从文件夹读取所有图片
    def get_all_image_lists(self, is_root_dir=True):
        # 得到的所有图片都存在set_images里。
        set_images = []
        # 获取当前目录下所有的子目录
        sub_dirs = [x[0] for x in os.walk(self.input_dir)]
        # 得到的第一个目录是当前目录，不需要考虑
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            # 获取当前目录下所有的有效图片文件。
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            for extension in extensions:
                file_glob = os.path.join(self.input_dir, dir_name, '*.'+extension)
                file_list.extend(glob.glob(file_glob))
            if not file_list:
                continue
            # 初始化当前类别的数据集
            for file_name in file_list:
                base_name = os.path.basename(file_name)
                set_images.append(base_name)
        # 返回整理好的所有数据
        print("the total number images is ", len(set_images))
        return set_images

    # 这个函数通过类别名称、所属数据集和图片编号获取一张图片的地址
    def get_image_path(self, img_all, index, input_dir=None):
        # 获取图片的文件名。
        base_name = img_all[index]
        # sub_dir = "wealthy_new_img"
        # 最终的地址为数据根目录的地址 + 类别的文件夹 + 图片的名称
        if not base_name.endswith('.jpg') or base_name.startswith('.'):
            return None
        else:
            if input_dir is None:
                full_path = os.path.join(self.input_dir, base_name)
            else:
                full_path = os.path.join(input_dir, base_name)
            return full_path

    # 获取图片对应的特征向量文件地址,图片地址后接".txt"
    def get_bottlenect_path(self, img_all, index):
        return self.get_image_path(img_all, index, self.output_dir) + '.txt'

    # 这个函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
    def get_or_create_bottleneck(self, sess, image_lists, index, jpeg_data_tensor, bottleneck_tensor, model_type):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        bottleneck_path = self.get_bottlenect_path(image_lists, index)
        # 如果这个特征向量文件不存在，则通过模型来计算特征向量，并将计算的结果存入文件。
        if not os.path.exists(bottleneck_path):
            # 获取原始的图片路径
            image_path = self.get_image_path(image_lists, index)
            # 获取图片内容。
            image_data_raw = gfile.FastGFile(image_path, 'rb').read()
            # 由于输入的图片大小不一致，此处得到的image_data大小也不一致（已验证），但却都能通过加载的模型生成一个2048的特征向量,模型包含预处理。
            # 通过模型计算特征向量
            try:
                if self.IS_PROCESSED is False:
                    image_temp = sess.run(self.img_process_tensor, feed_dict={self.img_input: image_data_raw})
                    self.image_data.append(image_temp[0])
                    self.out_bottleneck_path.append(bottleneck_path)
                    if len(self.image_data) < self.size_temp:
                        return
                if model_type == 'pb':
                    bottleneck_values = self.run_pb_bottleneck_on_image(sess, self.image_data, jpeg_data_tensor, bottleneck_tensor)
                else:
                    bottleneck_values = self.run_ckpt_bottleneck_on_image(sess, self.image_data, jpeg_data_tensor, bottleneck_tensor)
                for l in range(self.size_temp):
                    # 将计算得到的特征向量存入文件
                    if self.size_temp == 1:
                        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
                        with open(bottleneck_path, 'w') as bottleneck_file:
                            bottleneck_file.write(bottleneck_string)
                    else:
                        bottleneck_string = ','.join(str(x) for x in bottleneck_values[l])
                        with open(self.out_bottleneck_path[l], 'w') as bottleneck_file:
                            bottleneck_file.write(bottleneck_string)
                self.image_data = []
                self.out_bottleneck_path = []
            except Exception as e:
                print(e)
                print("error : %s " % image_path)
        # else:
        #     # 直接从文件中获取图片相应的特征向量。
        #     with open(bottleneck_path, 'r') as bottleneck_file:
        #         bottleneck_string = bottleneck_file.read()
        #     bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        # # 返回得到的特征向量
        # return bottleneck_values

    # 对批量指定的图片提取特征向量
    def create_batch_bottlenecks(self, sess, image_lists, num, jpeg_data_tensor, bottleneck_tensor, model_type):
        for index in range(num):
            self.get_or_create_bottleneck(sess, image_lists, index, jpeg_data_tensor, bottleneck_tensor, model_type)

    # 这个函数使用加载的训练好的pb模型处理一张图片，得到这个图片的特征向量
    def run_pb_bottleneck_on_image(self, sess, image_data, image_data_tensor, bottleneck_tensor):
        # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片的新的特征向量。
        bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
        # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    # 这个函数使用加载的训练好的ckpt模型处理一张图片，得到这个图片的特征向量。
    def run_ckpt_bottleneck_on_image(self, sess, image_data, image_data_tensor, bottleneck_tensor):
        bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
        # bottleneck_values = bottleneck_values['Predictions']
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values






