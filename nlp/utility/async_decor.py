from threading import Thread

def async_wrapper(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return wrapper