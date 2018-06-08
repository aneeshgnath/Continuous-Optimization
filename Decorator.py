def clock(func):
    
    import time

    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        return (result, elapsed)
    
    return clocked