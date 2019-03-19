import urllib.request
from io import BytesIO
import gzip
import re
import cv2
import os

def open_url(url):

    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    # try:
    #     html = response.read().decode("utf-8")
    # except:
    content = response.read()  # content是压缩过的数据
    buff = BytesIO(content)  # 把content转为文件对象
    f = gzip.GzipFile(fileobj=buff)
    html = f.read().decode('utf-8')
    return html

def get_image(html, index):
    p = r'<img class="wallpapers__image".*?src="([^"]*\.jpg)".*?>'
    imglist = re.findall(p, html)
    num = 1
    for each in imglist:
        # 读取图片数据
        response = urllib.request.urlopen(each)
        image = response.read()  # 不能进行'utf-8'编码,不能调用open_url()函数

        with open('./img/%s%s.jpg' % (index, num), 'wb') as fp:
            fp.write(image)
            print("正在下载第%s张图片" % num)
            num = num + 1
    return


if __name__ == '__main__':
    for i in range(10):
        url = "https://wallpaperscraft.com/all/page%s" % (i + 2)
        print(url)
        get_image(open_url(url), i)