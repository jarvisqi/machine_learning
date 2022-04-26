import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np


def sin():
    x = np.linspace(0, 10)
    print(x)
    y = np.sin(x)
    print(y)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="$sin(x)$", color="red", linewidth=1)

    plt.show()


def main():
    items=["aa","bb","cc","dd"]
    for  i,v in enumerate(items):
        print(i,v)
    
    print("\n")

    names = ['Bob', 'Alice', 'Guido']
    # 索引从1开始
    for index, value in enumerate(names, 1):
        print(f'{index}: {value}')


def shutil_module():
    """shutil 高级模块

    """
    # 拷贝文件
    # shutil.copyfile("C:/Users/XX/Downloads/1523599494692.jpg","C:/Users/XX/Pictures/9494692.jpg")
   
    #    压缩文件
    # shutil.make_archive("C:/Users/Jarvis/Downloads/math-5","zip",root_dir="C:/Users/Jarvis/Downloads/同济线性代数第五版")

    # # zip 压缩
    # with zipfile.ZipFile("C:/Users/Jarvis/Downloads/线性代数.zip",mode="w") as z:
    #     z.write('C:/Users/Jarvis/Downloads/mathext/同济线性代数第五版.pdf',arcname="线性代数.pdf")
    #     z.write('C:/Users/Jarvis/Downloads/mathext/线性代数知识网络图.pdf',arcname="线性代数知识.pdf")
    # zip 解压
    # with zipfile.ZipFile("C:/Users/Jarvis/Downloads/math-5.zip", mode="r") as z:
    #     z.extractall(path="C:/Users/Jarvis/Downloads/mathext")

    # 递归删除
    shutil.rmtree("C:/Users/Jarvis/Downloads/mathext",ignore_errors=True)
    # 重命名
    # shutil.move("C:/Users/Jarvis/Downloads/mlp","C:/Users/Jarvis/Downloads/nlp")
  

if __name__ == '__main__':
    #   sin()

    shutil_module()
