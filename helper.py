from time import time
  
  
def timer_func(name:str):
    # This function shows the execution time of 
    # the function object passed
    def inner_func(func):
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            print(f'Method {name} executed in {(t2-t1):.4f}s')
            return result
        return wrap_func
    return inner_func
