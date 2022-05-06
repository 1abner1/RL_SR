import numpy as np
from PIL import Image

img = Image.open(r'C:\Users\xww\Desktop\1.jpg').convert('RGBA')
arr = np.array(img)

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape) # 把图片转换为向量形式

print("打印图像数组", arr2)


# make a PIL image
img2 = Image.fromarray(arr2, 'RGBA')

# img2 = Image.fromarray(arr2 * 3, 'RGBA')
# print(246 * 3 - 256 * 2) # 超过255会自动减去256

img2.show()
img2.save('out.png') # 保存图片

arr2
