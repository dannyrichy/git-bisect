from time import time

from config import TIME_FLAG


def timer_func(name:str):
    """
    Decorator function to note down the time taken to execute

    :param name: Name of the function that is to be noted
    :type name: str
    """
    def inner_func(func):
        def wrap_func(*args, **kwargs):
            if TIME_FLAG:
                t1 = time()
                result = func(*args, **kwargs)
                print(f'Method {name} executed in {(time()-t1):.4f}s')
            else:
                result = func(*args, **kwargs)
            return result
        return wrap_func
    return inner_func
