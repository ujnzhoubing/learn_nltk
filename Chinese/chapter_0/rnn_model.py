# https://blog.csdn.net/chinatelecom08/article/details/83654602
import tensorflow as tf
import numpy as np


# ====================================搭建模型===================================
class RNNModel():
    # =======预定义模型参数========
    #VOCAB_SIZE = len(vocab)
    VOCAB_SIZE = None
    EPOCHS = 50
    BATCH_SIZE = 8
    TIME_STEPS = 100
    # BATCH_NUMS = len(numdata) // (BATCH_SIZE * TIME_STEPS)
    BATCH_NUMS = None
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 5
    MAX_GRAD_NORM = 1
    learning_rate = 0.003
    numdata = None

    def __init__(self, numdata, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS,
                 VOCAB_SIZE, learning_rate):
        super(RNNModel, self).__init__()
        self.numdata = numdata
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.HIDDEN_LAYERS = HIDDEN_LAYERS
        self.VOCAB_SIZE = VOCAB_SIZE

        self.BATCH_NUMS = len(numdata) // (BATCH_SIZE * self.TIME_STEPS)

        # ======定义占位符======
        with tf.name_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, None])
            self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, None])
            self.keepprb = tf.placeholder(tf.float32)

        # ======定义词嵌入层======
        with tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])
            emb_input = tf.nn.embedding_lookup(embedding, self.inputs)
            emb_input = tf.nn.dropout(emb_input, self.keepprb)

        # ======搭建lstm结构=====
        with tf.name_scope('rnn'):
            lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, state_is_tuple=True)
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keepprb)
            cell = tf.contrib.rnn.MultiRNNCell([lstm] * HIDDEN_LAYERS)
            self.initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell, emb_input, initial_state=self.initial_state)

        # =====重新reshape输出=====
        with tf.name_scope('output_layer'):
            outputs = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
            w = tf.get_variable('outputs_weight', [HIDDEN_SIZE, VOCAB_SIZE])
            b = tf.get_variable('outputs_bias', [VOCAB_SIZE])
            logits = tf.matmul(outputs, w) + b

        # ======计算损失=======
        with tf.name_scope('loss'):
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.targets, [-1])],
                                                                           [tf.ones([BATCH_SIZE * self.TIME_STEPS],
                                                                                    dtype=tf.float32)])
            self.cost = tf.reduce_sum(self.loss) / BATCH_SIZE

        # =============优化算法==============
        with tf.name_scope('opt'):
            # =============学习率衰减==============
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, self.BATCH_NUMS, 0.99, staircase=True)

            # =======通过clip_by_global_norm()控制梯度大小======
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), self.MAX_GRAD_NORM)
            self.opt = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, trainable_variables))

        # ==============预测输出=============
        with tf.name_scope('predict'):
            self.predict = tf.argmax(logits, 1)

    def save_mode(self, model, path='saved_model/'):
        # 保存模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('logs/tensorboard', tf.get_default_graph())

            sess.run(tf.global_variables_initializer())
            for k in range(self.EPOCHS):
                state = sess.run(model.initial_state)
                train_data = self.data_generator(self.numdata, self.BATCH_SIZE, self.TIME_STEPS)
                total_loss = 0.
                for i in range(self.BATCH_NUMS):
                    xs, ys = next(train_data)
                    feed = {model.inputs: xs, model.targets: ys, model.keepprb: 0.8, model.initial_state: state}
                    costs, state, _ = sess.run([model.cost, model.final_state, model.opt], feed_dict=feed)
                    total_loss += costs
                    if (i + 1) % 50 == 0:
                        print('epochs:', k + 1, 'iter:', i + 1, 'cost:', total_loss / i + 1)

            saver.save(sess, path+'lstm.ckpt')

        writer.close()

    # =======设计数据生成器=========
    def data_generator(self, data, batch_size, time_steps):
        samples_per_batch = batch_size * time_steps
        batch_nums = len(data) // samples_per_batch
        data = data[:batch_nums * samples_per_batch]
        data = data.reshape((batch_size, batch_nums, time_steps))
        for i in range(batch_nums):
            x = data[:, i, :]
            y = np.zeros_like(x)
            y[:, :-1] = x[:, 1:]
            try:
                y[:, -1] = data[:, i + 1, 0]
            except:
                y[:, -1] = data[:, 0, 0]
            yield x, y

    def test_model(self, id2char, path='saved_model/'):
        tf.reset_default_graph()
        evalmodel = RNNModel(1, self.HIDDEN_SIZE, self.HIDDEN_LAYERS, self.VOCAB_SIZE, self.learning_rate)
        # 加载模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, path+'lstm.ckpt')
            new_state = sess.run(evalmodel.initial_state)
            x = np.zeros((1, 1)) + 8
            samples = []
            for i in range(100):
                feed = {evalmodel.inputs: x, evalmodel.keepprb: 1., evalmodel.initial_state: new_state}
                c, new_state = sess.run([evalmodel.predict, evalmodel.final_state], feed_dict=feed)
                x[0][0] = c[0]
                samples.append(c[0])
            print('test:', ''.join([id2char[index] for index in samples]))