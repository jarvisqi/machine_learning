from builtins import sorted
from operator import itemgetter, attrgetter

users = {
    "Angelica": {
        "Blues	Traveler": 3.5,
        "Broken	Bells": 2.0,
        "Norah	Jones": 4.5,
        "Phoenix": 5.0,
        "Slightly	Stoopid": 1.5,
        "The	Strokes": 2.5,
        "Vampire	Weekend": 2.0
    },
    "Bill": {
        "Blues	Traveler": 2.0,
        "Broken	Bells": 3.5,
        "Deadmau5": 4.0,
        "Phoenix": 2.0,
        "Slightly	Stoopid": 3.5,
        "Vampire	Weekend": 3.0
    },
    "Chan": {
        "Blues	Traveler": 5.0,
        "Broken	Bells": 1.0,
        "Deadmau5": 1.0,
        "Norah	Jones": 3.0,
        "Phoenix": 5,
        "Slightly	Stoopid": 1.0
    },
    "Dan": {
        "Blues	Traveler": 3.0,
        "Broken	Bells": 4.0,
        "Deadmau5": 4.5,
        "Phoenix": 3.0,
        "Slightly	Stoopid": 4.5,
        "The	Strokes": 4.0,
        "Vampire	Weekend": 2.0
    },
    "Hailey": {
        "Broken	Bells": 4.0,
        "Deadmau5": 1.0,
        "Norah	Jones": 4.0,
        "The	Str okes": 4.0,
        "Vampire	Weekend": 1.0
    },
    "Jordyn": {
        "Broken	Bells": 4.5,
        "Deadmau5": 4.0,
        "Norah	Jones": 5.0,
        "Phoeni x": 5.0,
        "Slightly	Stoopid": 4.5,
        "The	Strokes": 4.0,
        "Vampire	Weekend": 4.0
    },
    "Json": {
        "Broken	Bells": 3.5,
        "Deadmau5": 3.0,
        "Norah	Jones": 4.5,
        "Phoeni x": 4.0,
        "Slightly	Stoopid": 2.5,
        "The	Strokes": 1.0,
        "Vampire	Weekend": 2.0
    },
    "Sam": {
        "Blues	Traveler": 5.0,
        "Broken	Bells": 2.0,
        "Norah	Jones": 3.0,
        "Phoe nix": 5.0,
        "Slightly	Stoopid": 4.0,
        "The	Strokes": 5.0
    },
    "Veronica": {
        "Blues	Traveler": 3.0,
        "Norah	Jones": 5.0,
        "Phoenix": 4.0,
        "Slig htly	Stoopid": 2.5,
        "The	Strokes": 3.0
    }
}


# print(users["Veronica"])
# 计算曼哈顿距离
class manhattan(object):
    def __init__(self):
        self.users = users

    def calc_manhattan(self, rating1, rating2):
        """
            计算曼哈顿距离。rating1和rating2参数中存储的数据格式均为				
            {'The	Strokes':	3.0,	'Slightly	Stoopid':	2.5}
        """
        distance = 0
        for key in rating1:
            if key in rating2:
                distance += abs(rating1[key] - rating2[key])
        return distance

    def getNearestUser(self, username):
        """
        计算所有用户和 username 用户的距离，
        倒序排列并返回结果列表
        """
        result = []
        for user in self.users:
            if user != username:
                distance = self.calc_manhattan(users[user], users[username])
                result.append((user, distance))
        # result.reverse() # 倒序
        # result.sort()  # 正序
        # sorted 可以使用内部元素排序
        #  reverse = True  降序 或者 reverse = False 升序
        result = sorted(result, key=lambda x: x[1], reverse=True)

        # 多级排序 operator 函数进行多级排序 
        # result = sorted(result, key=itemgetter(1, 0), reverse=True)

        return result

    def getrecommend(self, username: str, k=None) -> list:
        """
        返回相似的前 K 个物品
        """
        # 先找最进的用户
        neare = self.getNearestUser(username)[0][0]
        # 距离最近的用户评价过的
        neare_data = self.users[neare]
        # username 评价过的
        curr_data = self.users[username]
        recommand_result = []
        #  username 没有评价过的
        for artist in neare_data:
            if not artist in curr_data:
                recommand_result.append((artist, neare_data[artist]))
        # 根据评分排序
        result = sorted(recommand_result, key=lambda x: x[1], reverse=True)
        if k is not None:
            result = recommand_result[0:k]
        return result


if __name__ == '__main__':
    m = manhattan()
    # r = manhattan.calc_manhattan(users["Bill"], users["Sam"])
    # r1 = manhattan.calc_manhattan(users["Bill"], users["Veronica"])
    # print(r, r1)
    print(m.getrecommend("Angelica"))
