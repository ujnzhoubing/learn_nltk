# https://blog.csdn.net/chinatelecom08/article/details/83654602
import re
import numpy as np
from rnn_model import RNNModel
import tensorflow as tf


class LtsmGeneratePara:
    origon_file = "jinyong/shediaoyingxiongchuan_jinyong.txt"
    vocab = None
    numdata = None
    id2char = None

    def __init__(self):
        '''读取原始数据文件，并返回包含文件数据的一个list'''
        with open(self.origon_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            print(data[100])

        '''进行数据清洗'''
        # 将（）替换为空
        # 生成一个正则表达式，负责找（）包含的内容
        pattern = re.compile(r'\(.*\)')
        # 将（）替换为空
        data = [pattern.sub('', lines) for lines in data]
        print(data[100])

        # 将省略号替换为句号
        data = [line.replace('......', '。') for line in data if len(line) > 1]
        print(data[100])

        '''将每行的list合成一个长字符串'''
        data = ''.join(data)
        data = [char for char in data if self.is_uchar(char)]
        data = ''.join(data)
        print(data[:100])

        '''生成字典和一个字符的list'''
        vocab = set(data)
        self.vocab = vocab
        id2char = list(vocab)
        self.id2char = id2char
        char2id = {c: i for i, c in enumerate(id2char)}
        print('字典长度:', len(char2id))

        '''转换数据为数字格式'''
        numdata = [char2id[char] for char in data]
        numdata = np.array(numdata)
        self.numdata = numdata
        print('数字数据信息：\n', numdata[:10])
        print('\n文本数据信息：\n', ''.join([id2char[i] for i in numdata[:10]]))

    # ==============判断char是否是乱码===================
    def is_uchar(self, uchar):
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
        """判断一个unicode是否是数字"""
        if uchar >= u'\u0030' and uchar <= u'\u0039':
            return True
        """判断一个unicode是否是英文字母"""
        if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
            return True
        if uchar in ('，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'):
            return True
        return False


if __name__ == '__main__':
    ltsm = LtsmGeneratePara()

    # =======预定义模型参数========
    VOCAB_SIZE = len(ltsm.vocab)
    EPOCHS = 50
    BATCH_SIZE = 8
    TIME_STEPS = 100
    BATCH_NUMS = len(ltsm.numdata) // (BATCH_SIZE * TIME_STEPS)
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 5
    MAX_GRAD_NORM = 1
    learning_rate = 0.003

    model = RNNModel(ltsm.numdata, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE, learning_rate)
    model.save_mode(model)
    # ============模型测试============
    tf.reset_default_graph()
    evalmodel = RNNModel(ltsm.numdata, 1, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE, learning_rate)
    # 加载模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'saved_model/lstm.ckpt')
        new_state = sess.run(evalmodel.initial_state)
        x = np.zeros((1, 1)) + 8
        samples = []
        for i in range(100):
            feed = {evalmodel.inputs: x, evalmodel.keepprb: 1., evalmodel.initial_state: new_state}
            c, new_state = sess.run([evalmodel.predict, evalmodel.final_state], feed_dict=feed)
            x[0][0] = c[0]
            samples.append(c[0])
        print('test:', ''.join([ltsm.id2char[index] for index in samples]))


