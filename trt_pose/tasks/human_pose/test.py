import os
from os import listdir
from os.path import isfile, isdir, join

# 指定要列出所有檔案的目錄
mypath = "./video/"

# 取得所有檔案與子目錄名稱
files = listdir(mypath)

# 以迴圈處理
for f in files:
  # 產生檔案的絕對路徑
#   fullpath = join(mypath, f)
  # 判斷 fullpath 是檔案還是目錄
    # print(f)
    file_name = os.path.splitext(f)[0]
    print(file_name)
#   elif isdir(fullpath):
#     print("dir：", f)