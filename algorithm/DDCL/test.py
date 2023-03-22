def a1():
    x1 = 1
    print(x1)
    return x1

def b1():
    x2 = a1
    print(x2)
    return x2
print("开始执行")
v = b1

