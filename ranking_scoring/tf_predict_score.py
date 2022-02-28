# Copyright 2018 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""TF Ranking sample code for LETOR datasets in LibSVM format.

WARNING: All data sets are loaded into memory in this sample code. It is
for small data sets whose sizes are < 10G.

A note on the LibSVM format:
--------------------------------------------------------------------------
Due to the sparse nature of features utilized in most academic datasets for
learning to rank such as LETOR datasets, data points are represented in the
LibSVM format. In this setting, every line encapsulates features and a (graded)
relevance judgment of a query-document pair. The following illustrates the
general structure:

<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]

For example:

1 qid:10 32:0.14 48:0.97  51:0.45
0 qid:10 1:0.15  31:0.75  32:0.24  49:0.6
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
0 qid:20 4:0.79  31:0.01  33:0.05  35:0.27
3 qid:20 1:0.42  28:0.79  35:0.30  42:0.76

In the above example, the dataset contains two queries. Query "10" has 3
documents, two of which relevant with grades 1 and 2. Similarly, query "20"
has 1 relevant document. Note that query-document pairs may have different
sets of zero-valued features and as such their feature vectors may only
partly overlap or not at all.
--------------------------------------------------------------------------

Sample command lines:

OUTPUT_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train.txt && \
VALI=tensorflow_ranking/examples/data/vali.txt && \
TEST=tensorflow_ranking/examples/data/test.txt && \
rm -rf $OUTPUT_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_libsvm_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_libsvm_py_binary \
--train_path=$TRAIN \
--vali_path=$VALI \
--test_path=$TEST \
--output_dir=$OUTPUT_DIR \
--num_features=136

You can use TensorBoard to display the training results stored in $OUTPUT_DIR.
"""

from absl import flags
from absl.testing import flagsaver

import numpy as np
import pandas as pd
import six
import tensorflow as tf
import tensorflow_ranking as tfr

flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("vali_path", None, "Input file path used for validation.")
flags.DEFINE_string("test_path", None, "Input file path used for testing.")
flags.DEFINE_string("output_dir", None, "Output directory for models.")

flags.DEFINE_integer("train_batch_size", 128, "The batch size for training.")  # 32
flags.DEFINE_integer("num_train_steps", 100000, "Number of steps for training.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ['1001', '1001'],
                  "Sizes for hidden layers.")

flags.DEFINE_integer("num_features", 1536, "Number of features per document.")  # 1536
flags.DEFINE_integer("list_size", 2, "List size used for training. (qid 的前几个)")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")

flags.DEFINE_string("loss", "pairwise_logistic_loss",
                    "The RankingLossKey for loss function.")

FLAGS = flags.FLAGS


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        del coord
        self.iterator_initializer_fn(session)


def example_feature_columns():
    """Returns the example feature columns."""
    feature_names = ["{}".format(i + 1) for i in range(FLAGS.num_features)]
    return {
        name: tf.feature_column.numeric_column(
            name, shape=(1,), default_value=0.0) for name in feature_names
    }


def load_libsvm_data(path, list_size):
    """Returns features and labels in numpy.array."""

    def _parse_line(line):
        """Parses a single line in LibSVM format."""
        tokens = line.split("#")[0].split()
        assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
        label = float(tokens[0])
        qid = tokens[1]
        kv_pairs = [kv.split(":") for kv in tokens[2:]]
        features = {k: float(v) for (k, v) in kv_pairs}
        return qid, features, label

    tf.logging.info("Loading data from {}".format(path))

    # The 0-based index assigned to a query.
    qid_to_index = {}
    # The number of docs seen so far for a query.
    qid_to_ndoc = {}
    # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
    # a shape of [num_queries, list_size]. We use list for each of them due to the
    # unknown number of quries.
    feature_map = {k: [] for k in example_feature_columns()}
    label_list = []
    total_docs = 0
    discarded_docs = 0
    with open(path, "rt") as f:
        for line in f:
            qid, features, label = _parse_line(line)
            if qid not in qid_to_index:
                # Create index and allocate space for a new query.
                qid_to_index[qid] = len(qid_to_index)
                qid_to_ndoc[qid] = 0
                for k in feature_map:
                    feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
                label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
            total_docs += 1
            batch_idx = qid_to_index[qid]
            doc_idx = qid_to_ndoc[qid]
            qid_to_ndoc[qid] += 1
            # Keep the first 'list_size' docs only.
            if doc_idx >= list_size:
                discarded_docs += 1
                continue
            for k, v in six.iteritems(features):
                assert k in feature_map, "Key {} not founded in features.".format(k)
                feature_map[k][batch_idx][doc_idx, 0] = v
            label_list[batch_idx][doc_idx] = label

    tf.logging.info("Number of queries: {}".format(len(qid_to_index)))
    tf.logging.info("Number of documents in total: {}".format(total_docs))
    tf.logging.info("Number of documents discarded: {}".format(discarded_docs))

    # Convert everything to np.array.
    for k in feature_map:
        feature_map[k] = np.array(feature_map[k])
    return feature_map, np.array(label_list)


def get_eval_inputs(features, labels):
    """Set up eval inputs in a single batch."""
    iterator_initializer_hook = IteratorInitializerHook()

    def _eval_input_fn():
        """Defines eval input fn."""
        features_placeholder = {
            k: tf.placeholder(v.dtype, v.shape) for k, v in six.iteritems(features)
        }
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensors((features_placeholder,
                                                labels_placeholder))
        iterator = dataset.make_initializable_iterator()
        feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_placeholder[k]: features[k] for k in features_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _eval_input_fn, iterator_initializer_hook


def make_score_fn():
    """Returns a groupwise score fn to build `EstimatorSpec`."""

    def _score_fn(unused_context_features, group_features, mode, unused_params,
                  unused_config):
        """Defines the network to score a group of documents."""
        with tf.name_scope("input_layer"):
            group_input = [
                tf.layers.flatten(group_features[name])
                for name in sorted(example_feature_columns())
            ]
            input_layer = tf.concat(group_input, 1)
            tf.summary.scalar("input_sparsity", tf.nn.zero_fraction(input_layer))
            tf.summary.scalar("input_max", tf.reduce_max(input_layer))
            tf.summary.scalar("input_min", tf.reduce_min(input_layer))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = tf.layers.batch_normalization(input_layer, training=is_training)
        # # 新增卷积等层
        # tf.layers.conv2d(cur_layer)

        for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
            cur_layer = tf.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.layers.batch_normalization(cur_layer, training=is_training)
            cur_layer = tf.nn.relu(cur_layer)
            tf.summary.scalar("fully_connected_{}_sparsity".format(i),
                              tf.nn.zero_fraction(cur_layer))
        cur_layer = tf.layers.dropout(
            cur_layer, rate=FLAGS.dropout_rate, training=is_training)
        logits = tf.layers.dense(cur_layer, units=FLAGS.group_size)
        return logits

    return _score_fn


def get_eval_metric_fns():
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    metric_fns.update({
        "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
        # tfr.metrics.RankingMetricKey.ARP,
        tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
    ]
    })
    return metric_fns


def test_or_predict(is_eval):
    features_test, labels_test = load_libsvm_data(FLAGS.test_path,
                                                  FLAGS.list_size)
    test_input_fn, test_hook = get_eval_inputs(features_test, labels_test)

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=FLAGS.learning_rate,
            optimizer="Adam")  # Adagrad  Adam

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn)

    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=FLAGS.group_size,
            transform_fn=None,
            ranking_head=ranking_head),
        config=tf.estimator.RunConfig(FLAGS.output_dir, save_checkpoints_steps=100))

    # True:测试集评估精度； False:预测未知点的排序评分
    if is_eval:
        estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])
    else:
        predictions = estimator.predict(input_fn=test_input_fn, hooks=[test_hook])
        predict_list = list(predictions)
        print(np.shape(predict_list))
        return predict_list


def predict_all():
    all_predictlist = list()
    # csv_path = './libsvm/baidu_wealthy_dataset/wealthy_db_test_filter.csv'
    # csv_path = './libsvm/baidu_wealthy_dataset/wealthy_db_test.csv'
    csv_path = './predict/test_pair1.csv'
    read_datas = pd.read_csv(csv_path, header=None)
    df_datas = read_datas.values[:, :]
    max_num = -99
    min_num = 99
    with flagsaver.flagsaver(
            train_path='',
            vali_path='',
            test_path='./libsvm/inception_resnet_baidu_dir/btlk1001/wealthy_test.txt',
            output_dir='./tmp/inception_resnet_fuse/btlk/wealthy_baidu',  # _bd
            num_train_steps=500):
        predicts = test_or_predict(False)  # True:test  False:predict

        for i in range(len(predicts[:])):
            if predicts[i][0] > max_num:
                max_num = predicts[i][0]
            if predicts[i][1] > max_num:
                max_num = predicts[i][1]
            if predicts[i][0] < min_num:
                min_num = predicts[i][0]
            if predicts[i][1] < min_num:
                min_num = predicts[i][1]
            all_predictlist.append([df_datas[i][0], predicts[i][0]])
            all_predictlist.append([df_datas[i][1], predicts[i][1]])
    result_data = np.array(all_predictlist)
    delta = max_num - min_num
    print("%s, %s, %s" % (max_num, min_num, delta))
    all_data = result_data
    # all_data = result_data[np.lexsort(result_data.T)]
    out_path = './data09/test_bd_pair1.csv'
    # filter_set = set
    cls_level1 = 0
    cls_level2 = 0
    cls_level3 = 0
    cls_level4 = 0
    for i in range(len(all_data[:])):
        data = all_data[i]
        # if data[0] in filter_set:
        #     continue
        # filter_set.add(data[0])
        with open(out_path, 'a') as writer:
            score = (float(data[1]) - min_num) / delta
            if score <= 0.3:
                cls_level1 += 1
            elif 0.3 < score <= 0.5:
                cls_level2 += 1
            elif 0.5 < score <= 0.7:
                cls_level3 += 1
            else:
                cls_level4 += 1
            writer.write(str(data[0]) + "," + str(score) + '\n')
    print("%d, %d, %d, %d" % (cls_level1, cls_level2, cls_level3, cls_level4))
    print("finished!")


# 将预测结果添加到csv文件中
def extend_predict_data(file, predict_rst, st_index, end_index):
    path = '../DataProcess/bd_csv/%s.csv' % file
    read_datas = pd.read_csv(path, header=None)
    df_datas = read_datas.values[st_index:end_index, :]
    out_list = []
    for l in range(len(predict_rst)):
        line = list()
        line.extend(df_datas[l])
        line.extend(predict_rst[l])
        out_list.append(line)
    out_csv = '../DataProcess/bd_csv/by_fuse/%s_predict.csv' % (file + str(end_index))
    np.savetxt(out_csv, np.mat(out_list), fmt='%s', delimiter=',')


def large_predict_run():
    # 大数据分成多个模块进行预测
    all_predictlist = list()
    for i in range(6):
        with flagsaver.flagsaver(
                train_path='',
                vali_path='',
                test_path='../DataProcess/bottleneck/bd_mobilenet_v1/wealthy_allpairs_libsvm%s.txt' % str(
                    (i + 1) * 30000),
                output_dir='./tmp/mobilenet_v01_fuse',  # _bd
                num_train_steps=500):
            # test or predict
            predicts = test_or_predict(False)  # True:test  False:predict
            all_predictlist.extend(predicts)
            print(np.shape(all_predictlist))
    extend_predict_data('wealthy_bd_all_pairs', all_predictlist, 0, len(all_predictlist))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # 预测整个大数据集的排序评分
    # large_predict_run()
    predict_all()


if __name__ == "__main__":
    tf.app.run()
