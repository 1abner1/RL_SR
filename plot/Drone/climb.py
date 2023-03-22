import requests
import re
# from lxml import etree

# 发送 GET 请求
url = 'https://movie.douban.com/top250'
response = requests.get(url)

# 检查响应状态码，200 表示请求成功
if response.status_code == 200:
    # 使用正则表达式解析 HTML 内容，获取所有电影条目
    pattern = r'<span class="title">([\s\S]*?)</span>'
    matches = re.findall(pattern, response.text)

    # 打印每个电影的排名和名称
    for i, title in enumerate(matches):
        print(f'{i + 1}. {title}')
else:
    print('请求失败：', response.status_code)
