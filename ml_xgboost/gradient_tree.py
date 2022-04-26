# -*- coding:utf-8 -*-
import math


# 数据集，只包含两列
test_list = [[1, 5.56], [2, 5.7], [3, 5.81], [4, 6.4], [
    5, 6.8], [6, 7.05], [7, 7.9], [8, 8.7], [9, 9], [10, 9.05]]
step = 1 #eta
# 起始拆分点
init = 1.5
# 最大拆分次数
max_times = 10
# 允许的最大误差
threshold = 1.0e-3

def train_loss(t_list):
    sum = 0
    for fea in t_list:
        sum += fea[1]
    avg = sum * 1.0 / len(t_list)
    sum_pow = 0
    for fea in t_list:
        sum_pow = math.pow((fea[1]-avg), 2)
    return sum_pow, avg


def boosting(data_list):
    ret_dict = {}
    split_num = init
    while split_num < data_list[-1][0]:
        pos = 0
        for idx, data in enumerate(data_list):
            if data[0] > split_num:
                pos = idx
                break
        if pos > 0:
            l_train_loss, l_avg = train_loss(data_list[:pos])
            r_train_loss, r_avg = train_loss(data_list[pos:])
            ret_dict[split_num] = [ pos, l_train_loss+r_train_loss, l_avg, r_avg]
        split_num += step
    return ret_dict



def main():
    
    ret_list = []
    data_list = sorted(test_list, key=lambda x:x[0])

    time_num = 0
    while True:
        time_num += 1
        print('before split:',data_list)
        ret_dict = boosting(data_list)
        t_list = sorted(ret_dict.items(), key=lambda x:x[1][1])
        print('split node:',t_list[0])

        ret_list.append([t_list[0][0], t_list[0][1][1]])
        if ret_list[-1][1] < threshold or time_num > max_times:
            break
        for idx, data in enumerate(data_list):
            if idx < t_list[0][1][0]:
                data[1] -= t_list[0][1][2]
            else:
                data[1] -= t_list[0][1][3]
        print('after split:',data_list)

    print('split node and loss:')
    print('\n'.join(["%s\t%s" %(str(data[0]), str(data[1])) for data in ret_list]))


if __name__ == '__main__':
    main()
