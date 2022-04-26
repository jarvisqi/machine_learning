from threading import Thread
import time


class WorkerThread(Thread):

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None

    def run(self):
        self._return = self._target(*self._args, **self._kwargs)  
    
    def join(self):  
        Thread.join(self)  
        return self._return 


def call():
    result = []
    time.sleep(3)
    for i in range(10000):
        result.append(i)
    add = sum(result)
    return add


if __name__ == '__main__':

    print('程序运行。。。')

    worker = WorkerThread(target=call)
    worker.start()
    result = worker.join()

    print('程序结束:',result)
