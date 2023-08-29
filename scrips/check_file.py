import os
 
text = " "
path = "/home/zhao/DataSets/s3dis/points"
flag = 0
lister = []
def getfiles(path):
    # 将flag设置全局变量，防止递归查找时信息提示错误。
    global flag
    # 到给定的路径下
    os.chdir(path)
    # 返回一个列表，列表中的元素为path下的文件和子文件夹
    files = os.listdir()
    # 依次遍历
    for file_name in files:
        # 返回文件或子文件夹的绝对路径
        abs_path = os.path.abspath(file_name)
        # 如果为文件则进行查找
        f = open(abs_path,"rb")
        print(f)
        # if os.path.isfile(abs_path):
        #     # f = open(file_name, "rb")
        #     # 在文件中找到要查到的字符串后进行打印，并将flag置为1
        #     if text in f.read():
        #         flag = 1
        #         print(text + " found in ")
        #         print(abs_path)
            # else:
                # print("在{}中没查到！".format(abs_path))
                # lister.append(file_name)
                # print(lister)
        # 如果为文件夹则重复前面的动作
    #     if os.path.isdir(abs_path):
    #         getfiles(abs_path)
 
    # if flag == 0:
    #     print(text + " not found! ")
    #     return False
    # return True
 
 
 
# def delete_files():                                           #定义函数名称
#     for foldName, subfolders, filenames in os.walk(path):     #用os.walk方法取得path路径下的文件夹路径，子文件夹名，所有文件名
#         for filename in filenames:                         #遍历列表下的所有文件名
#             for data in lister:
#                 if filename == data:                       #当文件名不为“aaa.txt”时
#                     os.remove(os.path.join(foldName, filename))    #删除符合条件的文件
#                     print("{} deleted.".format(filename))           ##输出提示
 
getfiles(path)
# delete_files()         #调用定义的函数，注意名称与定义的函数名一致