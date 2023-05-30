from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import csv


A = []
B = []
#can phai mang khoe, on dinh trong qua trinh Scrapy
for i in range(2, 102):
    #link
    a = 'https://bonbanh.com/oto/ford-sf000000040/page,' + str(i)
    page = requests.get(a)
    souped = BeautifulSoup(page.content, "html.parser",from_encoding='utf-8')

    imgs = souped.find_all('img', {'class': 'h-car-img'})

    # lưu lại các ảnh trong thư mục
    count = 0
    count1 = 0
    for img in tqdm(imgs):
        imgData = img['src']
        if imgData == "img/s_noimage.gif":
            count1 = 1
            continue
        image = requests.get(imgData).content
        count = count + 1
        t = (i - 2) * 20 + count
        filename = r"download_ford" + '/' + str(t) + '.jpg'
        with open(filename, "wb") as file:
            file.write(image)
    sleep(0)
    if count1 == 1:
        continue

    #tim kiem hang xe
    elements1 = souped.find_all('h3', {'itemprop': 'name'})
    for element in elements1:
        name1 = element.text.split(' ')[0]
        A.append(name1)


    #tim kiem mau xe
    elements2 = souped.find_all('div', {'class': 'cb6_02'})
    for element in elements2:
        name2 = element.contents[0].split(',')[1].strip()
        B.append(name2)

print(len(A), len(B), t)

import matplotlib.pyplot as plt
from collections import Counter
'''''''''
# hiển thị biểu đồ đường dạng cột và giá trị của hãng xe
label_counts1 = Counter(A)
for label, count in label_counts1.items():
    print(f"{label}: {count}")

plt.bar(range(len(label_counts1)), list(label_counts1.values()), align='center')
plt.xticks(range(len(label_counts1)), list(label_counts1.keys()))
plt.show()
'''''''''
# hiển thị biểu đồ đường dạng cột và giá trị của màu xe
label_counts2 = Counter(B)
for label, count in label_counts2.items():
    print(f"{label}: {count}")

plt.bar(range(len(label_counts2)), list(label_counts2.values()), align='center')
plt.xticks(range(len(label_counts2)), list(label_counts2.keys()))
plt.show()




#tao file csv:
# Tạo DataFrame từ 2 mảng
df = pd.DataFrame({'Color': B, 'Mode': A})

# Lưu DataFrame thành file CSV
df.to_csv('file.csv3', index=False)

'''''''''
#ghi tiếp và file csv:
df = pd.read_csv('file.c    sv')
# Thêm dữ liệu vào DataFrame
new_data = {'Color': B, 'Mode': A}
new_df = pd.DataFrame(new_data)
df = df.append(new_df, ignore_index=True)

# Ghi DataFrame vào file CSV đã có
with open('file.csv', mode='a', newline='') as file:
    df.to_csv(file, index=False)
'''''''''