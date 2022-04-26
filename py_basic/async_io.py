import asyncio


async def read():
    sem = asyncio.Semaphore(5)
    with (await sem):  
        with open('./data/aw.txt') as f:
            content =  f.read()
    await asyncio.sleep(3)
    print("11111111111")


async def send():
    print("222222222222")
    return [1,2,3,4]


from threading import Thread
from time import sleep

def async_warpper(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return wrapper

@async_warpper
def A():
    sleep(20)
    print("a function")

def B():
    print("b function")
    return 101


def run():
    A()
    d = B()
    print(d)


if __name__ == '__main__':
    # main()

    # r1=read()
    # s1=send()
    # tasks = [
    #     asyncio.ensure_future(r1),
    #     asyncio.ensure_future(s1)
    # ]
 
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(asyncio.wait(tasks))

    run()
