import tensorflow as tf
import numpy as np
import jieba
import collections

stop_word = []
with open('F:\python_project\\tfword2vec_with_nec\stop_words.txt', encoding='utf-8') as f:
    line = f.readline()
    while line:
        stop_word.append(line)
        line = f.readline()
    stop_word = set(stop_word)

all_words = []
sentence_list = []
#file_dir = 'F:\\to_do_list.txt'
file_dir = 'F:\python_project\\tfword2vec_with_nec\\test.txt'

with open(file_dir) as file:
    line = file.readline()
    while line:
        word_list = []
        if '\n' in line:
            line = line.replace('\n', '')
        if '，' in line:
            line = line.replace('，', '')
        if ' ' in line:
            line = line.replace(' ', '')
        if len(line) > 0:
            word_line = list(jieba.cut(line))
            for word in word_line:
                if word not in stop_word and word not in ['qingkan520','www','com','http']:
                #if word not in stop_word:
                    all_words.append(word)
                    word_list.append(word)
        line = file.readline()
        sentence_list.append(word_list)
counter = collections.Counter(all_words)
word_dict = counter.most_common(30000)
dict_list = [x[0] for x in word_dict]
dictionary = {}
for i in range(len(dict_list)):
    dictionary[dict_list[i]] = i

step = 1
window_size = step * 2 + 1
data = []
label = []

for queue in sentence_list:
    for i in range(len(queue)):
        start = max(0, i - step)
        end = min(len(queue), i + step)
        for j in range(start, end):
            if j == i:
                continue
            elif j > i:
                data_tmp = dictionary.get(queue[i])
                label_tmp = dictionary.get(queue[j])
            else:
                data_tmp = dictionary.get(queue[i])
                label_tmp = dictionary.get(queue[j])
            if not (data_tmp and label_tmp):
                continue
            data.append(data_tmp)
            label.append(label_tmp)
data = np.array(data, dtype=np.int32)
label = np.array(label, dtype=np.int32)
label = np.reshape(label, [len(label), 1])
data_input = tf.placeholder(tf.int32, [len(data)])
data_output = tf.placeholder(tf.int32, [len(data), 1])
dict = {data_input:data, data_output:label}

vocab_size = 30000
embedding_size = 200
word_dict_embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
train = tf.nn.embedding_lookup(word_dict_embedding, data_input)
weight = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]))
bias = tf.Variable(tf.zeros([vocab_size]))
x = tf.shape(data_input)
y = tf.shape(data_output)
loss_tmp = tf.nn.nce_loss(weight, bias, data_output, train, 100, vocab_size)

loss = tf.reduce_mean(loss_tmp)
with tf.name_scope('nce_loss'):
    tf.summary.scalar('loss', loss)

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

merge = tf.summary.merge_all()
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('F:\python_project\log\word2vec', graph=sess.graph)
    for i in range(10000):
        summary, _, loss_word = sess.run([merge, train_op, loss], feed_dict=dict)
        if i % 100 == 0:
            print(loss_word)
            writer.add_summary(summary, i/100)



