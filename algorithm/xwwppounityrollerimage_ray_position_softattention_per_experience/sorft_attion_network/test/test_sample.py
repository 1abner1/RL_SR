prices = {
    'A': 123,
    'B': 450.1,
    'C': 480.1,
    'E': 490,
}

# 获得price最大的商品的key：
x=max(zip(prices.values(),prices.keys()))
print("x111111111111111111111",x)
print("x的类型",type(x))
y =x[1]
print("y值",y)
print("y值类型为",type(y))