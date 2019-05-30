from nltk.book import *


# 查找文本中使用最为频繁的前50个词
fdist1 = FreqDist(text1)
# print(fdist1)
vocabulary1 = list(fdist1.keys())
print(sorted(set(vocabulary1[:50])))

fdist1.plot(50, cumulative=True)

print(fdist1.hapaxes())