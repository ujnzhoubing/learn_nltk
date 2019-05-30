# 访问自己的语料库
import nltk
from nltk.corpus import PlaintextCorpusReader
corpus_root_after_fenci = "../chapter_0/jinyong_fenci/"
wordlists = PlaintextCorpusReader(corpus_root_after_fenci, ".*") # 读取语料库路径下的所有文件，包括子目录下的文件
# fileids实际上就是文件名
fileids = wordlists.fileids()
print(fileids)
# 访问指定文件的原始内容
sdyxz_str = wordlists.raw('shediaoyingxiongchuan_jinyong.txt')
sdxl_str = wordlists.raw('shendiaoxialv_jinyong.txt')
tlbb_str = wordlists.raw('tianlongbabu_jinyong.txt')
yttlj_str = wordlists.raw('yitiantulongji_jinyong.txt')
# nltk中的text对象接受一个list做为输入，而sdyxz_str是一个字符串，所以要进行分割为list对象
sdyxz = nltk.text.Text(sdyxz_str.split(' '))
sdxl = nltk.text.Text(sdyxz_str.split(' '))
tlbb = nltk.text.Text(tlbb_str.split(' '))
yttlj = nltk.text.Text(yttlj_str.split(' '))


# 搜索文本
sdyxz.concordance("郭靖")

# 近义词
sdyxz.similar("郭靖")

# 随机产生文本

