import time
import logging

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logging.debug('func:{} args:[{}, {}] took: {} sec'.format(f.__name__, args, kw, te-ts))
        return result
    return timed
