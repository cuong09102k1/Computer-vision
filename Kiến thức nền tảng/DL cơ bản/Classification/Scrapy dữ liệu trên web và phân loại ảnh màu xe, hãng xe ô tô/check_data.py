import pandas as pd
from matplotlib import pyplot as plt

'''''''''
df1 = pd.read_csv('file.csv1')
df2 = pd.read_csv('file.csv2')
df3 = pd.read_csv('file.csv3')
df4 = pd.read_csv('file.csv')

merged_df = pd.concat([df1, df2, df3, df4])
merged_df.to_csv('merged_file.csv', index=False)
'''''''''

df = pd.read_csv('merged_file.csv')
#print(df['Color'])

#so luong các màu
color_counts = df["Color"].value_counts()

#truc quan du lieu
from collections import Counter
sorted_counts = color_counts.sort_values()
# hiển thị biểu đồ đường dạng cột các màu
label_counts1 = Counter(df['Color'])
for label, count in label_counts1.items():
    print(f"{label}: {count}")

plt.barh(sorted_counts.index, sorted_counts.values)
plt.xlabel("Số lượng")
plt.ylabel("Màu")
plt.show()

'''''''''
#Gom cac label cho Color(để dễ trail)
#Label1: mau trang + mau bac + mau ghi + mau kem + mau xam
df['Color'].replace('màu trắng', 'label1', inplace=True)
df['Color'].replace('màu bạc', 'label1', inplace=True)
df['Color'].replace('màu ghi', 'label1', inplace=True)
df['Color'].replace('màu kem', 'label1', inplace=True)
df['Color'].replace('màu xám', 'label1', inplace=True)

#Label2: mau den
df['Color'].replace('màu đen', 'label2', inplace=True)

#Label3: mau do + mau nau + mau cam + mau vang + mau cat + mau dong
df['Color'].replace('màu đỏ', 'label3', inplace=True)
df['Color'].replace('màu nâu', 'label3', inplace=True)
df['Color'].replace('màu cam', 'label3', inplace=True)
df['Color'].replace('màu vàng', 'label3', inplace=True)
df['Color'].replace('màu cát', 'label3', inplace=True)
df['Color'].replace('màu đồng', 'label3', inplace=True)

#Label4: mau xanh + các màu ko xác định
df['Color'].replace('màu xanh', 'label4', inplace=True)
df['Color'].replace('máy xăng 4.0 L', 'label4', inplace=True)
df['Color'].replace('máy xăng 3.0 L', 'label4', inplace=True)
df['Color'].replace('máy xăng', 'label4', inplace=True)
df['Color'].replace('máy xăng 4.7 L', 'label4', inplace=True)
df['Color'].replace('máy xăng 1.5 L', 'label4', inplace=True)
df['Color'].replace('máy dầu 2.2 L', 'label4', inplace=True)
df['Color'].replace('máy dầu 2.0 L', 'label4', inplace=True)
df['Color'].replace('máy dầu 3.2 L', 'label4', inplace=True)
df['Color'].replace('màu hồng', 'label4', inplace=True)
df['Color'].replace('màu tím', 'label4', inplace=True)

# lưu lại file
df.to_csv('file_new.csv', index=False)
'''''''''
df_new = pd.read_csv('file_new.csv')
#truc quan du lieu
color_counts_new = df_new["Color"].value_counts()
from collections import Counter
sorted_counts_new = color_counts_new.sort_values()
# hiển thị biểu đồ đường dạng cột các màu
label_counts1_new = Counter(df_new['Color'])
for label, count in label_counts1_new.items():
    print(f"{label}: {count}")
plt.barh(sorted_counts_new.index, sorted_counts_new.values)
plt.xlabel("Số lượng")
plt.ylabel("Màu")
plt.show()

