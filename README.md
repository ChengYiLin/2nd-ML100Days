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

## Day6 : EDA與Outlier檢查

### 甚麼是例外值 (outlier)
<b>異常值 (Outliers) 出現的可能原因 : <br></b>
<ol>
	<li>可能的錯誤紀錄/手誤/系統性錯誤</li>
	<li>所以未知值，隨意填補 ( 例如:年齡 = -1 或 999 )</li>
</ol>

<b>檢查 Outliers 的流程與方法 : <br></b>
<ul>
	<li>盡可能確認每⼀個欄位的意義</li>
	<li>透過檢查數值範圍 (五值、平均數及標準差) 或繪製散點圖 (scatter)、分布圖 (histogram) 或其他圖檢查是否有異常</li>
</ul>

<b>對 Outliers 的處理⽅法 : <br></b>
<ul>
	<li>新增欄位⽤以紀錄異常與否</li>
	<li>填補 (取代)</li>
	<li>視情況以中位數, Min, Max 或平均數填補(有時會⽤ NA)</li>
</ul>

### Ways to detect and remove the Outliers
<a href="https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba">Reference : Ways to detect and remove the Outliers</a>
<ul>
	<li>作圖 : box plot (視覺呈現那些是異常值)</li>
	<li>作圖 : scatter plot (視覺呈現那些是異常值)</li>
	<li>統計 : Z-score ( Z score 是形容數值距離母群體平均值的距離，Z score 越大或小，就是 outlier</li>
	<li>統計 : IQR score ( 他的算法是將四分級距法的<b>Q3(75%)-Q1(25%)</b>得到 IQR score ，再去找小於 Q1-1.5*IQR 及 大於 Q3+1.5*IQR 的數值量)</li>
</ul>

#### Z score
```python
from scipt import stats
import numpy as np

z = np.abs(stats.zscore(data))
```

#### IQR score
```python
Q3 = data.quantile(0.75)
Q1 = data.quantile(0.25)

IQR = Q3-Q1

Outliers = data[ (data<(Q1 - 1.5 * IQR)) or ((Q3 + 1.5 * IQR)) ]
```

### Homework 補充 : ECDF (Empirical Cumulative Distribution Function)
![](https://cdn-images-1.medium.com/max/1200/0*pavjOIiT3R_GWDdz)

這種作圖方式可以顯示每筆數據的多寡及其相對應的百分位點，如上圖所示，我們先將數據由小到大排好，再將它的<b>排名除以總數</b>得到它的百分位，x 軸為我們的數值累加的結果(cumsum)，y軸為百分位(ECDF)。

### Day7 : 常用的數值取代：中位數與分位數連續數值標準化

### 如何處理例外值
在上一章我們已經知道如何找出例外值(outlier)，處理例外值的方法就是將它以其他數值做填補，避免影響到母群體的結構

<table>
	<tr style="background: LightCyan; text-align: center;">
		<th>常用以填補的統計值</th>
		<th>方法</th>
	</tr>
	<tr>
		<td>中位數(median)</td>
		<td>np.median(value_array)</td>
	</tr>
	<tr>
		<td>平均數(mean)</td>
		<td>np.mean(value_array)</td>
	</tr>
	<tr>
		<td>分位數(quantile)</td>
		<td>np.quantile(value_arrar, q = ...)</td>
	</tr>
	<tr>
		<td>眾數(mode)</td>
		<td>dictionary method: 較快的方法<br>scipy.stats.mode(value_array): 較慢的方法</td>
	</tr>
</table>

### 連續型數值標準化

![](https://i.imgur.com/bVgoFqn.png)

<b>為什麼要標準化?</b><p>因為不同單位可能會影響模型產出的結果，例如 1000 公克 vs 1 公斤 ； 1美元 vs 30元台幣 等</p>
<b>一定要標準化嗎?</b><p>不一定，看你的模型是哪一種，例如 Regression model: 有差 ； Tree-based model: 沒差 等</p>

<table>
	<tr style="background: LightCyan; text-align: center;">
		<th>常用的標準化方法</th>
		<th>方法</th>
	</tr>
	<tr>
		<td>Z轉換</td>
		<td><img src="https://i.imgur.com/xlU3s8V.png" alt="z-value"></td>
	</tr>
	<tr>
		<td>空間壓縮</td>
		<td><p>Y 為輸出的空間範圍</p><img src="https://i.imgur.com/4y1D7tJ.png" alt="空間壓縮"></td>
	</tr>
</table>
<p style="font-weight: bolder;">特殊狀況 : </p>
有時候我們不會使用 min/max ⽅法進行標準化，⽽會採用 Qlow/Qhigh normalization (min = q1 ; max = q99)

### 是否每次資料都要進行標準化呢?
有的時候好，有得時候不好，為什麼?
<ol>
	<li>Good</li>
	<ol>
		<li>某些演算法 (如 SVM, DL) 等，對權重敏感或對損失函數平滑程度有幫助者</li>
		<li>特徵間的量級差異甚大</li>
	</ol>
	<li>Bad</li>
	<ol>
		<li>有些指標性的單位是不適合做 normalization，如 錢的多寡</li>
		<li>量的單位在某些特徵上是有意義的，對於 model 來說是重要的</li>
	</ol>
</ol>



### Homework 補充 : collection 的 defaultdict 及 Counter

#### defaultdict
我們在使用 python dictionary 的時候，要調用key值來操作的時候，都要先看看 key 值是否存在，不然就會出現 key Error，遇到這種狀況過去我們只能乖乖地先給 dictionary key 值，再做操作，如下方程式碼
```python
word_list = ['a','b','x','a','a','b','z']
counter = {}
for element in word_list:
    if element not in counter:
        counter[element] = 1
    else:
        counter[element] += 1
```
<p>那麼我們就不能直接先給 dictionary 全部給一個 default 值 0，就直接給它 += 1，不用再去 if/else 判斷嗎?</p>
其實是可以的，collection 這套件中的 defaultdict 就解決這個問題省去我們的麻煩，但我們在宣告 defaultdict 給它一個 default 值時，我們需要餵它一個 <font color="red">Function</font>

```python
from collections import defaultdict

better_dict = defaultdict(int) # default值以一個 int() function 產生

word_list = ['a','b','x','a','a','b','z']

for element in word_list:
    better_dict[element]+=1
```

function 我們可以自行去定義，也可使用 python 內建的

```python
from collections import defaultdict

def zero(): return 0  # 小提醒: 也可寫成 zero = lambda:0

counter_dict = defaultdict(zero) # default值以一個zero()方法產生
a_list = ['a','b','x','a','a','b','z']

for element in a_list:
        counter_dict[element] += 1

print(counter_dict) # 會輸出defaultdict(<function zero at 0x7fe488cb7bf8>, {'x': 1, 'z': 1, 'a': 3, 'b': 2})
```

其他小例子
```python
from collections import defaultdict

multi_dict = defaultdict(list) 
key_values = [('even',2),('odd',1),('even',8),('odd',3),('float',2.4),('odd',7)]

for key,value in key_values:
    multi_dict[key].append(value)

print(multi_dict) # 會輸出defaultdict(<class 'list'>, {'float': [2.4], 'even': [2, 8], 'odd': [1, 3, 7]})
```
#### Counter
除了上述的 defaultdict 之外, collection 還有一個方便我們去做 count 的資料結構 <b>Counter</b>，下方是它的例子
```python
from collections import Counter

a_str = 'abcaaabccabaddeae'
counter = Counter(a_str) # 可直接由初始化的方式統計個數
print(counter)
print(counter.most_common(3)) # 輸出最常出現的3個元素
print(counter['z']) # 對於不存在的key值給出default值0
counter.update('aaeebbc') # 可用update的方式繼續統計個數
```

<p>參考 :</p>
<ul>
	<li> <a href="https://ithelp.ithome.com.tw/articles/10193094">30天python雜談</a></li>
	<li> <a href="https://docs.python.org/3.7/library/collections.html#collections.defaultdict">python 官方</a> </li>
</ul>