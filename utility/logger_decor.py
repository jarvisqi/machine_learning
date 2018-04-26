import logging
import time


def create_logger():
    """
    创建日志
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    isotimeformat = '%Y-%m-%d'
    filename = time.strftime(isotimeformat, time.localtime(time.time()))
    handler = logging.FileHandler("./logs/{}.txt".format(filename))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def exception(logger):
    """
    异常装饰器
    @param logger: The logging object
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                err = "There was an exception in  "
                err += func.__name__
                logger.exception(err)

        return wrapper

    return decorator

