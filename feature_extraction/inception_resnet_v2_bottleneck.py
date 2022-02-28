from tensorflow.python.platform import gfile
import tensorflow as tf
import time
import sys
sys.path.append('../../StreetViewScore/')
# 导入网络，CKPT模型用
import model_process.inference_bottleneck as infetence_bottleneck
import model_process.nets.inception_resnet_v2 as model
slim = tf.contrib.slim

# 待处理图片目录
INPUT_DATA = "/home/tensorflow/LWPWORK/StreetViewScore/dataCenter/google/placepulse/"
# 谷歌图片 "/home/tensorflow/LWPWORK/StreetViewScore/dataCenter/google/placepulse/"
# 标注的百度图片  "/home/tensorflow/LWPWORK/StreetViewScore/dataCenter/baidu/imgs/"
# 所有百度图片    "/home/tensorflow/LWPWORK/StreetViewScore/dataCenter/baidu/imgs/"

# 输出特征向量目录；涵盖 btlk、avgpool、predict三类
OUT_DIR_BOTTLENECK = "bottleneck/inception_resnet_google_dir/predict1001"

# 图像输入大小
INPUT_SIZE = [1, 299, 299, 3]      # 批量输入可减少时间
# 输出结果
OUTPUT_SIZE = 1001
# 是否需预处理图片
IS_PROCESSED = False
BATCH = 100

# CKPT模型文件路径
CHECKPOINT_FILE = 'ckpt/inception_resnet_v2.ckpt'

"""该部分配置仅PB模型用到"""
# PB模型文件路径
PB_MODEL_FILE = "pb/tensorflow_inception_graph.pb"
# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  #'input:0'
# 模型瓶颈层张量所对应的名称
# BOTTLENECK_TENSOR_NAME = 'InceptionResnetV2/Logits/Logits/BiasAdd:0'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# BOTTLENECK_TENSOR_NAME = 'InceptionResnetV2/Logits/AvgPool_1a_8x8/AvgPool:0'    # 1536
# 最终预测层前一层1001张量    'InceptionResnetV2/Logits/Logits/BiasAdd:0'         # 1001
# 最终预测层张量   'InceptionResnetV2/Logits/Logits/Predictions:0'                # 1001
# 瓶颈层张量    'pool_3/_reshape:0'                                               # 1001





def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph..
    Args:
      input_width: Desired width of the image fed into the recognizer graph.
      input_height: Desired width of the image fed into the recognizer graph.
      input_depth: Desired channels of the image fed into the recognizer graph.
      input_mean: Pixel value that should be zero in the image for the graph.
      input_std: How much to divide the pixel values by before recognition.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)

    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def pb_bottleneck(bottleneckCenter, img_all):
    total_num = len(img_all)
    # 读取已经训练好的模型。
    with gfile.FastGFile(PB_MODEL_FILE, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 加载读取的模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量。
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                          JPEG_DATA_TENSOR_NAME], name='')
    # 指定某部分图片提取特征值
    # image_list = image_list[28000:40000]
    # total_num = 12000
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 训练过程
        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名才开始进队
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        temp_list = []
        time0 = time.time()
        time1 = time0
        for i in range(total_num):
            # 每次获取一个batch的训练数据；30张图片一个线程
            temp_list.append(img_all[i])
            if (i % BATCH == 0 and i > 0) or (i + 1 == total_num):
                bottleneckCenter.create_batch_bottlenecks(sess, temp_list, len(temp_list), jpeg_data_tensor, bottleneck_tensor, 'pb')
                temp_list = []
            if i % 1000 == 0:
                print("the number of bottleneck: %s , time %s: " % (i, time.time() - time1))
                time1 = time.time()
        print("the number of bottleneck: %s , total time %s: " % (total_num, time.time() - time0))
        # 终止线程
        coord.request_stop()
        coord.join(threads)


def ckpt_bottleneck(bottleneckCenter, img_all):
    total_num = len(img_all)
    # 读取已经训练好的ckpt模型。
    arg_scope = model.inception_resnet_v2_arg_scope()
    # 创建网络
    with slim.arg_scope(arg_scope):
        # 输入图片
        jpeg_data_tensor = tf.placeholder(dtype=tf.float32, shape=INPUT_SIZE)
        # 创建网络
        logits, end_points = model.inception_resnet_v2(jpeg_data_tensor, is_training=False, num_classes=OUTPUT_SIZE)
        bottleneck_tensor = end_points['Logits']
        #['PreLogitsFlatten'] ['Logits']
        params = slim.get_variables_to_restore(exclude=[])
        # 'InceptionResnetV2/AuxLogits/Logits', 'InceptionResnetV2/Logits/Logits'
        # 用于恢复模型  如果使用这个保存或者恢复的话，只会保存或者恢复指定的变量
        restorer = tf.train.Saver(params)
        # 用于保存检查点文件
        save = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.latest_checkpoint("")  # train_log_dir  微调后ckpt模型路径
            if ckpt is not None:
                save.restore(sess, ckpt)
                print('从微调保存后的模型加载！')
            else:
                restorer.restore(sess, CHECKPOINT_FILE)  # CHECKPOINT_FILE 官方ckpt文件地址
                print('从官方模型加载！')
            # 创建一个协调器，管理线程
            coord = tf.train.Coordinator()
            # 启动QueueRunner, 此时文件名才开始进队。
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            temp_list = []
            time0 = time.time()
            time1 = time0
            for i in range(total_num):
                # 每次获取一个batch的训练数据
                temp_list.append(img_all[i])
                if (i % BATCH == 0 and i > 0) or (i + 1 == total_num):
                    bottleneckCenter.create_batch_bottlenecks(sess, temp_list, len(temp_list), jpeg_data_tensor, bottleneck_tensor, 'ckpt')
                    temp_list = []
                if i % 1000 == 0:
                    print("the number of bottleneck: %s , time %s: " % (i, time.time() - time1))
                    time1 = time.time()
            print("the number of bottleneck: %s , total time %s: " % (total_num, time.time() - time0))
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    img_input, img_one_tensor = add_jpeg_decoding(299, 299, 3, 127.5, 127.5)
    # 创建瓶颈向量管理器
    bottleneckCenter = infetence_bottleneck.BottleneckCenter(INPUT_DATA, OUT_DIR_BOTTLENECK, INPUT_SIZE, IS_PROCESSED,
                                                             img_input, img_one_tensor)
    # 读取所有图片
    image_all = bottleneckCenter.get_all_image_lists(False)
    # 利用PB模型获取特征向量
    # pb_bottleneck(bottleneckCenter, image_all)
    # 利用CKPT模型获取特征向量
    ckpt_bottleneck(bottleneckCenter, image_all)
