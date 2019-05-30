# 主要目的：使用jieba对中文语料库进行分词

#  这段代码写的并不好，只是利用了nltk读取了一下文件名而已
from nltk.corpus import PlaintextCorpusReader
corpus_root_before_fenci = "jinyong/" # 指定未分词中文语料库的根路径
wordlists = PlaintextCorpusReader(corpus_root_before_fenci, ".*") # 读取语料库路径下的所有文件，包括子目录下的文件
# fileids实际上就是文件名
fileids = wordlists.fileids()
print(type(fileids))


# 使用jieba进行中文分词

import jieba
corpus_root_after_fenci = "jinyong_fenci/" # 指定分词后中文词料库的存储路径
for file_name in fileids:
    w = jieba.cut(open(corpus_root_before_fenci + file_name, encoding='utf-8').read())
    wlst = list(w)  # 得到分词列表
    w1 = " ".join(wlst)  # 得到空格划分的分词后文本字符串
    f = open(corpus_root_after_fenci+file_name, "w", encoding="utf-8")
    f.write(w1)
    f.close()

'''
import jieba
corpus_root_after_fenci = "jinyong_fenci/" # 指定分词后中文词料库的存储路径
for file_name in fileids:
    w = jieba.cut(open(corpus_root_before_fenci + file_name, encoding='utf-8').read())
    wlst = list(w)  # 得到分词列表
    f = open(corpus_root_after_fenci+file_name, "w", encoding="utf-8")
    for word in wlst:
        f.write(word + '\n')
    f.close()
'''

