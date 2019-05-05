# 2nd ML 100 Days
###### tags: `Python` `ML 100 days`

## Day 1 : 資料介紹與評估指標

**Machine Learning 分為兩類,分別為**
* 監督式學習
* 非監督式學習

![](https://i.imgur.com/9VIyb9v.jpg)

### 面對資料我們應該思考哪些問題?
* **為什麼這個問題重要？**
    * 例如: 企業的核⼼心問題、公眾利益、影響政策方向等。
* **資料從何而來？**
    * 例如: 網頁爬蟲、網站流量、購物車紀錄等。
* **資料的型態是什麼？**
    * data、image、vedio、text ......
* **我們可以回答什麼問題？**
    * 通常是以一些數學評估的指標來衡量,例如: 正確率、Mean absolute error等

## Day 2 : EDA-1/讀取資料
### 甚麼是 EDA ? 
**EDA(Exploratory Data Analysis)** 透過 **視覺化/統計工具**進行分析，達到三個主要目的
* 了解資料
* 發現 outliers 或異常數值
* 分析各變數間的關聯性

我們可以根據 EDA 的結果調整分析的方向，可以在模型建立之前，發現潛在的錯誤

### 數據分析的流程

![](https://i.imgur.com/9XIGH1J.jpg)

## Day 3 : 如何新建一個 dataframe ?

### 為什麼新建 dataframe 重要 ?
* 需要把分析過程中所產生的數據或者結果儲存為結構化的資料
* 測試程式碼，統一資料結構方便視覺化呈現


### 如何建立 dataframe ?

**Way1 : ( Use Dictionary to build )**
```typescript='py'
import pandas as pd
data = {'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
        'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
        'visitor': [139, 237, 326, 456]}
        
visitors_1 = pd.DataFrame(data)
```

**Way2 : ( Use list and zipped )**
```typescript='py'
cities = ['Austin', 'Dallas', 'Austin', 'Dallas']
weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
visitors = [139, 237, 326, 456]

list_labels = ['city', 'weekday', 'visitor']
list_cols = [cities, weekdays, visitors]

zipped = list(zip(list_labels, list_cols))

visitors_2 = pd.DataFrame(dict(zipped))
```

### 讀取及寫入資料
#### <font color=red>csv 檔</font>
<font color=#483D8B>**讀取 :**</font>
```python
import pandas as pd
data = pd.read_csv('path_of_file/file.csv')
```
<font color=#483D8B>**寫入 :**</font>
```python
import pandas as pd
df.to_csv('path_of_file/file.csv',encoding='utf-8',index=False)
```

#### <font color=red>txt 檔</font>
<font color=#483D8B>**讀取 :**</font>
```python
with open("path_of_file/example.txt", 'r') as f:
    data = f.readlines()
print(data)
```
<font color=#483D8B>**寫入 :**</font>
```python
number = ['One\n','Two\n','Three\n','Four\n','Five\n']

with open('../test.txt','w') as f:
    f.write("This is write by \'f.write \' \n")
    f.writelines(number)
    f.close()
```

#### <font color=red>json 檔</font>
json 檔是一種 javascript 傳送檔案的形式，類似於 python 的 dictionary

<font color=#483D8B>**讀取 :**</font>
```python
import json
with open('path_of_file/example01.json', 'r') as f:
    j1 = json.load(f)
j1
```
<font color=#483D8B>**寫入 :**</font>
```python
import json
df.to_json('path_of_file/example01.json')
```

#### <font color=red>npy 檔</font>
npy 是 numpy 以二進位制格式儲存的檔案名稱

<font color=#483D8B>**讀取 :**</font>
```python
import numpy as np
array_back = np.load('path_of_file/example.npy')
array_back
```
<font color=#483D8B>**寫入 :**</font>
```python
import numpy as np
data = np.arange(10)
np.save('path_you_want_to_save/name_of_file',data)
```

#### <font color=red>pickle 檔</font>
pickle 是 python 一種保存、壓縮、提取檔案的一種二進位制格式，以 dictionary 的形式處存，和 json 檔的不同之處在於 pickle 可以使用 python 的一些內建 class 的用法

參考 : https://docs.python.org/zh-cn/3/library/pickle.html#pickle-inst

<font color=#483D8B>**讀取 :**</font>
```python
import pickle
with open('path_you_want_to_save/example.pkl', 'rb') as f:
    pkl_data = pickle.load(f)
pkl_data
```
<font color=#483D8B>**寫入 :**</font>
```python
data = {'Hello':'World'}

import pickle
with open('../test.pkl','wb') as f:
    pickle.dump(data,f)
```

<font color='red'>**註:**</font> 這邊在 r 後面加一個 b 是因為檔案是二進位制格式(binary)

### 讀取圖片

讀取圖片的方法目前有以下幾種
* skimage
* PIL
* OpenCV

一開始我們必須先匯入套件來呈現圖片(numpy、matplotlib)
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

#### <font color=red>skimage</font>
```python
import skimage.io as skio
img1 = skio.imread('../data/Part01/example.jpg')
plt.imshow(img1)
plt.show()
```

#### <font color=red>PIL</font>
```python
from PIL import Image
# 這時候還是 PIL object
img2 = Image.open('../data/Part01/example.jpg') 
img2 = np.array(img2)
plt.imshow(img2)
```

#### <font color=red>OpenCV</font>
```python
import cv2
img3 = cv2.imread('../data/Part01/example.jpg')
plt.imshow(img3)
plt.show()

# 因為opencv 的顏色順序是 BGR
# 所以我們必須把它改成 RGB 的順序，圖片才是正常色

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
plt.imshow(img3)
plt.show()
```

## Day4 : EDA資料類型介紹

拿到資料的第一步，通常就是看我們有什麼，觀察有什麼欄位，這些欄位代表什麼意義、以什麼樣的資料類型來儲存

### 了解 pandas dataframe 欄位的基本資料類型
* 資料的欄位變數一般可分成**離散變數**及**連續變數**
    * **離散變數** : 用整數單位去計算的量(國家、性別、數量等)
    * **連續變數** : 一定區間內可以任意取值的變數(身高、時間、車速等)
* dataframe 的資料類形大致上分為 3 種
    * int64 ( 可以是 離散變數 或 連續變數 )
    * float64 ( 可以是 離散變數 或 連續變數 )
    * object (字串 或 類別變數)
    
    資料類型還包括 boolean、date 等等


資料若是字串、類別的資料形式，我們通常在分析時都會將它轉換一下型態，比較好來做分析，轉換方式有以下兩種
* Label encoding (通常是有序的，如年齡分 老人、中年、小孩)
* One Hot encoding (通常是無序的，如國家)

### 查看欄位
```python
# 檢視資料中各個欄位類型的數量
import pandas as pd
dataframe.dtypes.value_counts()

# 檢視資料中類別型欄位各自類別的數量
dataframe.select_dtypes(include=["object"]).apply(pd.Series.nunique, axis = 0)
```

### Label encoding
這通常在處理偏向有順序類別的 column ，如年齡族群 老中少 這類的
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dataframe['col'] = le.transform(dataframe['col'])
```

### One hot encoding
處理無順序的類別(如顏色、國家、性別等)
```python
import pandas as pd

dataframe = pd.get_dummies(dataframe)
```

## Day5 : EDA資料分佈

### 統計量化的方式
* 計算集中趨勢
	* 平均值 Mean
	* 中位數 Median
	* 眾數 Mode 

* 計算資料分散程度
	* 最小值 Min 、 最大值 Max
	* 範圍 Range
	* 四分位差 Quartiles 
	* 變異數 Variance
	* 標準差 Standard deviation

```python
import pandas as pd

dataframe.describe()
```

<b>相關參考連結 : </b>
<a href="http://www.hmwu.idv.tw/web/R_AI_M/AI-M1-hmwu_R_Stat&Prob.pdf">吳漢銘老師AI學校經理人班教材</a>

### 常用的視覺化套件

* <a href="https://matplotlib.org/gallery/index.html">matplotlib</a>

* <a href="https://seaborn.pydata.org/examples/index.html">seaborn</a>