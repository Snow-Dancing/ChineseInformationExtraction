import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')


def save2log(string):
    with open("./log.txt", 'a', encoding='utf8') as f:
        f.write(string + '\n')
    print(string)



def main(_):
    # the path to save models
    save_path = './model/'
    save_epoch_path = './epochnum.txt'
    save2log('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')
    # 训练数据
    save2log('reading training data')
    train_y = np.load('./data/train_y.npy')
    train_word = np.load('./data/train_word.npy')
    train_pos1 = np.load('./data/train_pos1.npy')
    train_pos2 = np.load('./data/train_pos2.npy')
    # 测试数据
    test_y = np.load('./data/testall_y.npy')
    test_word = np.load('./data/testall_word.npy')
    test_pos1 = np.load('./data/testall_pos1.npy')
    test_pos2 = np.load('./data/testall_pos2.npy')


    #print(train_word[0])
    #print(000)
    # print(len(train_word[1]))
    # print(111)
    # os._exit(1)
    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])
    test_len = int(len(test_word) / float(settings.big_num))

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.0005)
            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            saver = tf.train.Saver()
            if os.path.isfile(save_epoch_path):
                with open(save_epoch_path, 'r', encoding='utf8') as f:
                    tem = f.read().split()
                one_epoch = int(tem[0]) + 1
                step = tem[1]
                saver.restore(sess, save_path + 'ATT_GRU_model-' + step)
            else:
                sess.run(tf.global_variables_initializer())
                one_epoch = 1

            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num, is_train=True):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch
                if is_train:
                    temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                        [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                        feed_dict)
                else:
                    step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                        [global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                        feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)

                if is_train and step % 50 == 0:
                    tempstr = "{} - step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                    save2log(tempstr)
                if not is_train:
                    return loss, acc

            while one_epoch <= settings.num_epochs:
                save2log("开始第%d个epoch" % one_epoch)
                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)
                for i in range(int(len(temp_order) / float(settings.big_num))):

                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        save2log('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)

                    train_step(temp_word, temp_pos1, temp_pos2, temp_y, len(temp_word))

                    current_step = tf.train.global_step(sess, global_step)
                # 测试
                time_str = datetime.datetime.now().isoformat()
                save2log("{} - Evaluating...共{}个样本".format(time_str, len(test_word)))
                loss_count, acc_count = 0, 0
                for i in range(test_len):
                    temp_word = test_word[i * settings.big_num:(i + 1) * settings.big_num]
                    temp_pos1 = test_pos1[i * settings.big_num:(i + 1) * settings.big_num]
                    temp_pos2 = test_pos2[i * settings.big_num:(i + 1) * settings.big_num]
                    temp_y = test_y[i * settings.big_num:(i + 1) * settings.big_num]
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        save2log('out of range')
                        continue
                    temloss, temacc = train_step(temp_word, temp_pos1, temp_pos2, temp_y, len(temp_word), False)
                    loss_count += temloss
                    acc_count += temacc
                time_str = datetime.datetime.now().isoformat()
                tempstr = "{} - test : loss {:g}, acc {:.2f}".format(time_str, loss_count/test_len, acc_count/test_len)
                save2log(tempstr)
                # 保存模型
                save2log('saving model')
                path = saver.save(sess, save_path + 'ATT_GRU_model', global_step=current_step)
                tempstr = 'have saved model to ' + path
                save2log(tempstr)
                with open(save_epoch_path, 'w', encoding='utf8') as f:
                    f.write(str(one_epoch) + " " + str(current_step))
                one_epoch += 1


if __name__ == "__main__":
    tf.app.run()
