# -*- coding: utf-8 -*-
# 计算修正余弦相似度的Python代码


from math import sqrt

users3 = {
          "David": {"爱乐之城": 4, "荒野猎人": 5,"银河护卫队2": 4, "长城": 1},
          "Matt": {"爱乐之城": 3, "荒野猎人": 4,"银河护卫队2": 4, "长城": 1},
          "Ben": {"美国队长3": 4, "爱乐之城": 3,"银河护卫队2": 3, "长城": 1},
          "Chris": {"美国队长3": 3, "爱乐之城": 4,"荒野猎人": 4, "银河护卫队2": 3},
          "Tori": {"美国队长3": 5, "爱乐之城": 4,"荒野猎人": 5, "长城": 3}
          }

def computeSimilarity(band1, band2, userRatings):
    averages = {}
    for (key, ratings) in userRatings.items():
        averages[key] = (float(sum(ratings.values())) / len(ratings.values()))

    num = 0 # 分子
    dem1 = 0 # 分母的第一部分
    dem2 = 0
    for (user, ratings) in userRatings.items():
        if band1 in ratings and band2 in ratings:
            avg = averages[user]
            num += (ratings[band1] - avg) * (ratings[band2] - avg)
            dem1 += (ratings[band1] - avg) ** 2
            dem2 += (ratings[band2] - avg) ** 2
    return num / (sqrt(dem1) * sqrt(dem2))

print("美国队长3 和 银河护卫队2 相似度: %f " % (computeSimilarity('美国队长3', '银河护卫队2', users3)))
print("爱乐之城 和 银河护卫队2  相似度： %f " % (computeSimilarity('爱乐之城', '银河护卫队2', users3)))
print("荒野猎人 和 银河护卫队2  相似度： %f " % (computeSimilarity('荒野猎人', '银河护卫队2', users3)))
