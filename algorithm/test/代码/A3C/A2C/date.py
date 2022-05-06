import datetime
if __name__ == '__main__':
    start_time = datetime.datetime.now().second
    iter= 0
    while True:
        now_time = datetime.datetime.now().second
        print(now_time - start_time)
        iter += 1
        if (iter >= 100000):
            break