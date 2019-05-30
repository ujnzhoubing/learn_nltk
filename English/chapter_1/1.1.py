# 安装nltk
import nltk
# nltk.download() # 下载完成后就可以注释掉了

from nltk.book import *

print("\n查看text1的文本类型及内容")
print(type(text1))
print(text1)
# text1~text9为Text类的实例对象名称，它们都代表一本书籍。实际上Text类的构造函数接受一个单词列表作为参数，NLTK库预先帮我们构造了几个Text对象。

# 搜索文本,会忽略大小写
print("\n搜索文本")
text1.concordance("monstrous")

# 查找相似的词
print("\n查找相似的词")
text1.similar("monstrous")
text2.similar("monstrous")

# 使用两个或者两个以上词汇的上下文
print("\n使用两个或者两个以上词汇的上下文")
text2.common_contexts(["monstrous", "very"])

# 显示词在文章中的分布
print("\n显示词在文章中的分布")
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

# 随机产生一些文本
print("\n随机产生一些文本")
print(text3.generate("hairy"))

# 计数词汇
print("\n计数词汇")
print(len(text3))

# 排序词汇
print("\n排序词汇")
print(sorted(set(text3)))