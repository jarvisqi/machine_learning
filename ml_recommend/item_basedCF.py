# -*- coding: utf-8 -*-

import math


class item_bsedCF:

    def __init__(self, train_file):
        self.train_file = train_file
        self.readData()
        

    def readData(self):
        # 读取文件，并生成用户-物品的评分表和测试集  
        self.train = dict()  # 用户-物品的评分表  
        for line in open(self.train_file):
            # user,item,score = line.strip().split(",")  
            user, score, item = line.strip().split(",")
            self.train.setdefault(user, {})
            self.train[user][item] = int(float(score))

    def itemSimilarity(self):
        # 建立物品-物品的共现矩阵  
        C = dict()  # 物品-物品的共现矩阵  
        N = dict()  # 物品被多少个不同用户购买  
        for user, items in self.train.items():
            for i in items.keys():
                N.setdefault(i, 0)
                N[i] += 1
                C.setdefault(i, {})
                for j in items.keys():
                    if i == j: continue
                    C[i].setdefault(j, 0)
                    C[i][j] += 1
                    # 计算相似度矩阵  
        self.W = dict()
        for i, related_items in C.items():
            self.W.setdefault(i, {})
            for j, cij in related_items.items():
                self.W[i][j] = cij / (math.sqrt(N[i] * N[j]))
        return self.W

    # 给用户user推荐，前K个相关用户
    def recommend(self, user, K=3, N=10):
        rank = dict()
        action_item = self.train[user]  # 用户user产生过行为的item和评分  
        for item, score in action_item.items():
            for j, wj in sorted(self.W[item].items(), key=lambda x: x[1], reverse=True)[0:K]:
                if j in action_item.keys():
                    continue
                rank.setdefault(j, 0)
                rank[j] += score * wj
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:N])


if __name__ == '__main__':
    item = item_bsedCF("D:\\Learning\\Matlab\\data\\uid_score_bid")
    item.itemSimilarity()
    user = "xiyuweilan"
    print("推荐以下{0}个商品给 {1} 用户".format(5,user))
    redic = item.recommend(user, 3, 5)
    for k in redic:
        print(k, "\t", redic[k])
