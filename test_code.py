import time

def a(*args):
    print(len(args))


print(time.strftime('%Y%m%d%H%M-%S', time.localtime()))


a([1, 2, 3, 4], [7777],'123456')


