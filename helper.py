from time import time

from config import TIME_FLAG


def timer_func(name:str):
    # This function shows the execution time of 
    # the function object passed
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
