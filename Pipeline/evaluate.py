# encoding=utf8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import pickle
import json
import tensorflow as tf
from model import Model
from data_utils import load_word2vec, input_from_line
from utils import get_logger, load_config, create_model
import numpy as np
import time
import datetime
import network
from sklearn.metrics import average_precision_score
flags = tf.app.flags
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
FLAGS = tf.app.flags.FLAGS

def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122

def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger, False)
        while True:
            line = input("请输入测试句子:")
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            CRET(result)

def CRET(test_data):
    pathname = "./model/ATT_GRU_model-15000"
    wordembedding = np.load('./data/vec.npy')
    test_settings = network.Settings()
    test_settings.vocab_size = 16693
    test_settings.num_classes =49
    test_settings.big_num = 1
    schemas_path = './test/schemas.json'
    # save_path = './test/valid_final_result.json'
    # database = []
    # with open(test_path, 'r', encoding='utf8')as f:
    #     lines = f.readlines()
    # for line in lines:
    #     database.append(json.loads(line))
    # total_test = len(database)
    # print("一共有%d个测试样本" % total_test)

    with open(schemas_path, 'r', encoding='utf8') as f:
        schemas = json.load(f)

    # 定义一个相近的实体类别的字典
    similar_entity_dict = {"历史人物" : "人物",
                           "人物" : "历史人物",
                           "书籍" : "图书作品",
                           "图书作品" : "书籍",
                           "影视作品" : "电视综艺",
                           "电视综艺" : "影视作品"
                           }

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

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

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy


            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)
            saver.restore(sess, pathname)

            # print('reading word embedding data...')
            vec = []
            word2id = {}
            f = open('./data/vec.txt', encoding='utf-8')
            content = f.readline()
            content = content.strip().split()
            # dim = int(content[1])
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                word2id[content[0]] = len(word2id)
                content = content[1:]
                content = [(float)(i) for i in content]
                vec.append(content)
            f.close()
            word2id['UNK'] = len(word2id)
            word2id['BLANK'] = len(word2id)

            # print('reading relation to id')
            relation2id = {}
            id2relation = {}
            f = open('./data/relation2id.txt', 'r', encoding='utf-8')
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                relation2id[content[0]] = int(content[1])
                id2relation[int(content[1])] = content[0]
            f.close()

            ####################################################################
            sentence = "".join(test_data['string'].lower().split())
            entities_dict = {entity['word'] : entity['type'] for entity in test_data['entities']}
            couples = [(x,y) for x in entities_dict.keys() for y in entities_dict.keys() if x != y]
            new_couples = []
            for (x,y) in couples:
                if (y,x) not in new_couples:
                    new_couples.append((x,y))
            spo_list = []
            for (en1, en2) in new_couples:
                # en1, en2, sentence = line.strip().split()
                #print("实体1: " + en1)
                #print("实体2: " + en2)
                #print("句子：" + sentence)
                #relation = 0
                en1pos = sentence.find(en1)
                if en1pos == -1:
                    en1pos = 0
                en2pos = sentence.find(en2)
                if en2pos == -1:
                    en2post = 0
                output = []
                # length of sentence is 70
                fixlen = 70
                # max length of position embedding is 60 (-60~+60)
                maxlen = 60

                #Encoding test x
                for i in range(fixlen):
                    word = word2id['BLANK']
                    rel_e1 = pos_embed(i - en1pos)
                    rel_e2 = pos_embed(i - en2pos)
                    output.append([word, rel_e1, rel_e2])

                for i in range(min(fixlen, len(sentence))):

                    word = 0
                    if sentence[i] not in word2id:
                        #print(sentence[i])
                        #print('==')
                        word = word2id['UNK']
                        #print(word)
                    else:
                        #print(sentence[i])
                        #print('||')
                        word = word2id[sentence[i]]
                        #print(word)

                    output[i][0] = word
                test_x = []
                test_x.append([output])

                #Encoding test y
                label = [0 for i in range(len(relation2id))]
                label[0] = 1
                test_y = []
                test_y.append(label)

                test_x = np.array(test_x)
                test_y = np.array(test_y)

                test_word = []
                test_pos1 = []
                test_pos2 = []

                for i in range(len(test_x)):
                    word = []
                    pos1 = []
                    pos2 = []
                    for j in test_x[i]:
                        temp_word = []
                        temp_pos1 = []
                        temp_pos2 = []
                        for k in j:
                            temp_word.append(k[0])
                            temp_pos1.append(k[1])
                            temp_pos2.append(k[2])
                        word.append(temp_word)
                        pos1.append(temp_pos1)
                        pos2.append(temp_pos2)
                    test_word.append(word)
                    test_pos1.append(pos1)
                    test_pos2.append(pos2)

                test_word = np.array(test_word)
                test_pos1 = np.array(test_pos1)
                test_pos2 = np.array(test_pos2)


                prob, accuracy = test_step(test_word, test_pos1, test_pos2, test_y)
                prob = np.reshape(np.array(prob), (1, test_settings.num_classes))[0]
                #print("关系是:")
                # print(prob)
                top3_id = prob.argsort()[-3:][::-1]
                rel_id = prob.argsort()[-1:][::-1][0]
                #print(id2relation[rel_id] + ", Probability is " + str(prob[rel_id]))
                rel = id2relation[rel_id]
                prob = prob[rel_id]
                rel_dict = {}
                en1_type = entities_dict[en1]
                en2_type = entities_dict[en2]
                if prob > 0.9:
                    process_Result(en1, en2, en1_type, en2_type, schemas, rel_dict, rel)
                    if not rel_dict:
                        if en1_type in similar_entity_dict:
                            en1_type = similar_entity_dict[en1_type]
                            process_Result(en1, en2, en1_type, en2_type, schemas, rel_dict, rel)
                            if not rel_dict and en2_type in similar_entity_dict:
                                en2_type = similar_entity_dict[en2_type]
                                process_Result(en1, en2, en1_type, en2_type, schemas, rel_dict, rel)
                        elif en2_type in similar_entity_dict:
                            en2_type = similar_entity_dict[en2_type]
                            process_Result(en1, en2, en1_type, en2_type, schemas, rel_dict, rel)
                if rel_dict:
                    spo_list.append(rel_dict)
            print({"text":sentence, "spo_list":spo_list})



def process_Result(en1, en2, en1_type, en2_type, schemas, rel_dict, rel):
    if en1_type == schemas[rel]['object_type'] and en2_type == schemas[rel]['subject_type']:
        rel_dict['predicate'] = rel
        rel_dict['object_type'] = en1_type
        rel_dict['subject_type'] = en2_type
        rel_dict['object'] = en1
        rel_dict['subject'] = en2
    elif en2_type == schemas[rel]['object_type'] and en1_type == schemas[rel]['subject_type']:
        rel_dict['predicate'] = rel
        rel_dict['object_type'] = en2_type
        rel_dict['subject_type'] = en1_type
        rel_dict['object'] = en2
        rel_dict['subject'] = en1


def main(_):
    evaluate_line()

if __name__ == "__main__":
    tf.app.run(main)



