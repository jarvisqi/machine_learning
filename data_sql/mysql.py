#导入pymysql的包
import pymysql
import threading

def demo(conn):

    try:
        cur=conn.cursor()#获取一个游标
        cur.execute('select id from 10 ORDER BY RAND() LIMIT 1000')
        data=cur.fetchall()
        print(len(data))
        # cur.close()#关闭游标
        # conn.close()#释放数据库资源
    except  Exception :print("查询失败")


 def get_connect(self):
        """ 
        获取连接信息 
        返回: conn.cursor() 
        """
        # if not self.db:
        #    raise(NameError,"没有设置数据库信息")
        self.conn = pymysql.connect(host='10',user='10',passwd='10.com',db='10',port=10,charset='utf8mb4')
        cur = self.conn.cursor()
        if not cur:
            raise (NameError, "连接数据库失败")
        else:
            return cur


def loop():
    threads = []
    threads_num = 5  # 线程数量
        
    con1=get_connect()
    con2=get_connect()
    con3=get_connect()
    con4=get_connect()
    con5=get_connect()

    t1 = threading.Thread(target=demo, args=(con1))
    threads.append(t1)

    t2 = threading.Thread(target=demo, args=(con2))
    threads.append(t2)
    t3 = threading.Thread(target=demo, args=(con3))
    threads.append(t3)
    t4 = threading.Thread(target=demo, args=(con4))
    threads.append(t4)
    t5 = threading.Thread(target=demo, args=(con5))
    threads.append(t5)


if __name__ == '__main__':
    loop()
