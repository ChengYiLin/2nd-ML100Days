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

![](https://i.imgur.com/6xUcinV.jpg)

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

![](https://i.imgur.com/6aQyKGk.jpg)

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

## Day 8 : 常用的 DataFrame 操作
> Pandas 官方參考資料 : [link](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

### <font color=#4472C4>Creating Dataframe</font>
#### <font color='red'>One index</font> 
Way1 : Dictionary
```python
Name = ['Nick','Leo','Martin']
Age = [23,28,22]
Gender = ['M','F','M']

people = pd.DataFrame(
    {'Name':Name,
     'Age':Age,
     'Gender':Gender},
    index = ['a','b','c']
)
people
```

Way2 : Array
```python
people = pd.DataFrame(np.arange(9).reshape((3,3)),
                     index=['a','b','c'],
                     columns=['Col1','Col2','Col3'])
people
```

#### <font color='red'>MultiIndex</font>

```python
Name = ['Nick','Leo','Martin']
Age = [23,28,22]
Gender = ['M','F','M']

people = pd.DataFrame(
    {'Name':Name,
     'Age':Age,
     'Gender':Gender},
    index = pd.MultiIndex.from_tuples(
        [('B',1),('B',2),('A',3)],
        names=['Grade','Num']
    )
)
people
```

### <font color=#4472C4>Concatenate Dataframe</font>
Data : 
```python
# 生成範例用的資料 ()
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])
```

#### 將 data 接在下面 (沿縱軸加上去)
Way 1 : append
```python
df1.append(df3)
```

Way 2 : concat
```python
df_one_index = pd.concat([df1,df2])
df_multi_index = pd.concat([df1,df2],keys=['d1','d2'])
```

#### 將 data 接在旁邊 (沿橫軸加上去)
Use **concat**
```python
df = pd.concat([df1,df2],axis=1)
```

#### 條件合併 (inner、outer)
Use **merge**
```python
# 聯集
df = pd.merge(df1,df4,how='outer')

# 交集
df = pd.merge(df1,df4,how='inner')
```

### <font color=#4472C4>Reshaping Dataframe</font>
Data
```python
columns = ['name', 'age', 'gender', 'job']

user1 = pd.DataFrame(
	[['alice',19, "F", "student"], 
	['john', 26, "M", "student"],
	['eric', 22, "M", "student"],
	['paul', 58, "M", "manager"],
	['peter',33, 'M', 'engineer'],
	['julie',44, 'F', 'scientist']],
	columns=columns)
```

#### <font color='red'>melt</font>
melt 是將 dataframe 轉換成只有兩欄的 Dataframe，分別是  variable 及 value。

variable 是欄位名稱，value 是欄位的值，我們也可決定那些原本的欄位當 index

```python
# 我們可以用參數的調整決定那些欄位的去留，也可決定那些欄位要當 index
test = user1.melt(id_vars=['name'],value_vars='age',var_name='col')
test
```

#### <font color='red'>pivot</font>
pivot 則是 melt 的相反操作，將欄位的值轉換成欄位，所以它常和 melt 搭配操作，讓我們有效率的 reshape 我們的 dataframe

這邊特別要注意的是 pivot 參數記得要填寫，不像melt，不然就給你 Error
```python
test.pivot(index='name',columns='variable',values='value')
```

### <font color=#4472C4>Sorting in Dataframe</font>
```python
# sort value
df.sort_values('col_you_wanr_to_sort',ascending=False)

# sort index
df.sort_index()
```

### <font color=#4472C4>Subset</font>
```python
# 依條件篩選我們的 data ， 並顯示所有 columns
sub_df = df[df.column_name > 10]

# 
```

### <font color=#4472C4>Groupby</font>
Data :
```python
data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(data)
```

依 column 的種類去做分群
```python
group_data = df.groupby('Year')
for name,group in group_data:
    print(name)
    print(group)

#2014
#     Team  Rank  Year  Points
#0  Riders     1  2014     876
#2  Devils     2  2014     863
#4   Kings     3  2014     741
#9  Royals     4  2014     701
#2015
#       Team  Rank  Year  Points
# 1   Riders     2  2015     789
# 3   Devils     3  2015     673
# 5    kings     4  2015     812
# 10  Royals     1  2015     804
# 2016
#      Team  Rank  Year  Points
# 6   Kings     1  2016     756
# 8  Riders     2  2016     694
# 2017
#       Team  Rank  Year  Points
# 7    Kings     1  2017     788
# 11  Riders     2  2017     690
```
我們也可以選擇特定族群
```python
group_data.get_group(2014)

#     Team  Rank  Year  Points
#0  Riders     1  2014     876
#2  Devils     2  2014     863
#4   Kings     3  2014     741
#9  Royals     4  2014     701
```
我們可以針對特定族群套用特定的函數運算
```py
# 這邊我們選擇 Points 我們特定的 column，並使用 np.mean 運算
group_data.Points.agg(np.mean)

# 我們也可以套用多重的 函數
group_data.Points.agg([np.mean, np.std, np.sum])

# 我們可以使用 count 來計數、nunique 來算 unique值 的總數
group_data.Points.agg('count')
group_data.Points.agg('nunique')
```
我們可以自行設定 function 將數值轉換(放大、縮減甚麼的)
```python
group_data.get_group(2014)

# 我們定義 function 將數值 *2
score = lambda x:x*2

# 搭配我們的 transform ，將我們指定的 Points column 的值乘 2
group_data.get_group(2014).Points.transform(score)
```


### <font color=#4472C4>Other</font>
#### 重新命名欄位
```python
# rename 函數 用 dictionary 來改變 column 及 index 的名稱
df.rename(columns={"yeas_old":"age",'gender':'job'},index={'alice':'Alice'})
```

#### index的相關操作
```python
# 重設我們的 index
df.reset_index()

# 照我們的 index 排序
df.sort_index()
```

#### 去除特定的index
```python
# 去除 index (列)
df.drop(['index_name_1','index_name_2'])
```

#### 去除一整欄 column
```python
# 去除 column (行)
df.drop(columns=['columns_name_1','columns_name_2'])
```

#### 時間序列的好夥伴 stack、unstack
[參考資料](https://www.cnblogs.com/bambipai/p/7658311.html)
```python
# 堆疊
df.stack()

# 去除堆疊
df.unstack()
```

## Day 9 : correlation/相關係數簡介

### What is Correlation Coefficient?
相關係數是其中⼀個常用來了解各個欄位與我們想要預測的目標之間的關係指標。

相關係數是衡量**兩個隨機變量之間**線性關係的強度和方向

![](https://i.imgur.com/YdVE14p.png)

相關係數是一個介於 -1 ~ 1 之間的值，<br>
負值代表負相關，正值代表正相關，<br>
數值的大小代表相關的程度

![](https://i.imgur.com/0SXvHfG.png)

如果我們想了解兩個變數之間的 **線性關係** 時，<br>
能給出一個 -1 ~ 1 之間的量化關係

### Code
np.corrcoef( )
```py
# 使用 numpy 的 corrcoef 來算相關係數
np.corrcoef(x, y)

# Output : 
# array([[1.        , 0.02215266],
#        [0.02215266, 1.        ]])
```
除了得到數值外，我們也很常使用 scatter 來視覺化
```py
plt.scatter(x, y)
```

## Day 10 : EDA from Correlation
* **Correlation 小提醒**
  
我們在作圖的時候，如果其中一個 axis 是屬於種類的話 (如不是 1 就是 0，非連續型資料)

這時我們算 correlation 時就不能用一般的 pearson correlation，

而是要用 **point-biaseral correlation** 來進行處理

[Reference](https://blog.xuite.net/stat200635/Biostatistics/574342853-Point-biserial%E7%9B%B8%E9%97%9C%E4%BF%82%E6%95%B8+%28Point-biserial+correlation+coefficient%29)

```python
from scipy import stats

# pointbiserialr
stats.pointbiserialr(a, b)

# pearsonr
stats.pearsonr(a, b)
```

* **作圖小提醒**
  
![](https://i.imgur.com/fyhjUm6.png)

* **作圖小提醒**

我們可以取修改單位，或是用 log 去處理

![](https://i.imgur.com/tvkpKfJ.png)

## Day 11 : 繪圖與樣式 ＆ KDE (Kernel Density Estimation)

### <font color=#4472C4>繪圖樣式</font>
matplotlib 的繪圖風格可以主要有三種樣式
* default
* ggplot
* seaborn
  
**code** 
```python
import matplotlib.pyplot as plt

# 在畫圖前先設定
plt.style.use('default') 
plt.style.use('ggplot') 
plt.style.use('seaborn') 
```

### <font color=#4472C4>Kernal Density Estimation (KDE)</font>
KDE 是一種估計未知的密度函數，觀察變數的機率密度

它有以下幾個特點 :
* 用無母數方法，對分布沒有假設
* 計算量大，對電腦吃效能
* 在線下的面積總和為 1


當要算 KDE 時，有幾種常見的三種函數
1. Gaussian 
2. Cosine
3. Triangular

```python
# KDE, 比較不同的 kernel function，用 kernal 去做調整
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Gaussian esti.', kernel='gau')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Cosine esti.', kernel='cos')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Triangular esti.', kernel='tri')

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.show()
```
![](https://i.imgur.com/lVoyVpg.png)

### <font color=#4472C4>Distplot (將 bar 與 Kde 同時呈現)</font>

```python
sns.distplot(app_train.loc[app_train.TARGET==1,'DAYS_BIRTH'] / 365, label = 'target == 1')
plt.legend()
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.show()
```

![](https://i.imgur.com/DNQXshO.png)

### <font color=#4472C4>分群比較</font>
常常我們要分群做分析，例如 20 ~ 70 歲，我們要分組，那要怎麼做呢?

**舉例 :** 

先將 data 歸類，分好組別
```python
# 我們對於連續數值可以先用 linspace 分好，製造出陣列
# 20 ~ 70 歲，分 10 組，切 11 個點
bin_cut = np.linspace(20, 70, num = 11)

# 我們再用 pd.cut 去切，去分群
# pd.cut(你要切的欄位，bins = 剛剛所分好的陣列範圍 )
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = bin_cut)

# 顯示不同組的數量 
print(age_data['YEARS_BINNED'].value_counts())
```

使用 groupby 來做分群，分群做平均
```python
# 計算每個年齡區間的 Target、DAYS_BIRTH與 YEARS_BIRTH 的平均值
age_groups  = age_data.groupby('YEARS_BINNED').mean()
```
![](https://i.imgur.com/CKFzahE.png)

我們就可以做分群的 bar chart
```python
# 以年齡區間為 x, target 為 y 繪製 barplot
import seaborn as sns
plt.figure(figsize=(8,6))
px = age_groups.index
py = age_groups.TARGET*100
sns.barplot(px, py)

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
```
![](https://i.imgur.com/4ncqUJq.png)

## Day 12 : 把連續型變數離散化

連續型變數連續化的目的 : 
* 數據簡單化，選項變少了
* 離散型變數比較穩定，干擾比較小

### 常用離散劃分法
1. 等寬劃分 (依連續型 data 的範圍，等寬切幾等分)
2. 等頻劃分 (將資料分成幾等分，個等分的資料數量一樣)
3. 聚類劃分 (將資料依類別劃分，像 Day11 提到一樣)

#### code
```python
# 初始設定 Ages 的資料
ages = pd.DataFrame({"age": [18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})

# 新增欄位 "equal_width_age", 對年齡做 等寬劃分(pd.cut)
ages["equal_width_age"] = pd.cut(ages["age"], 4)

# 新增欄位 "equal_freq_age", 對年齡做 等頻劃分(pd.qcut)
ages["equal_freq_age"] = pd.qcut(ages["age"], 4)

# 觀察離散劃分下, 每個種組距各出現幾次
print(ages["equal_width_age"].value_counts())
print(ages["equal_freq_age"].value_counts())
```

## Day 13 : 程式實作(複習groupby 及 離散化)

**Hint** : 先離散化後，再用 groupby 製成分類

## Day 14 : Subplots
使用時機 : 
* 同一組資料，但想同時⽤用不同的圖型呈現
* 有很多相似的資訊要呈現時 (如不同組別的比較)

### example 
```py
# 每張圖大小為 8x8，先設定我們 figure 的大小
plt.figure(figsize=(8,8))

# plt.subplot 三碼如上所述, 分別表示 row總數, column總數, 本圖示第幾幅(idx)
plt.subplot(321)
plt.plot([0,1],[0,1], label = 'I am subplot1')
plt.legend()

plt.subplot(322)
plt.plot([0,1],[1,0], label = 'I am subplot2')
plt.legend()

plt.subplot(323)
plt.plot([1,0],[0,1], label = 'I am subplot3')
plt.legend()

plt.subplot(324)
plt.plot([1,0],[1,0], label = 'I am subplot4')
plt.legend()

plt.subplot(325)
plt.plot([0,1],[0.5,0.5], label = 'I am subplot5')
plt.legend()

plt.subplot(326)
plt.plot([0.5,0.5],[0,1], label = 'I am subplot6')
plt.legend()


# 呈現
plt.show()
```

## Day 15 : Heatmap & Grid-plot

### Heat map 
Heatmap 常用於呈現訊息的強弱 (以顏⾊色深淺呈現)，也常用於呈現混淆矩陣

* 常用於呈現變數間的相關性
* 也可以用於呈現不同條件下，數量的高低關係

#### example
```py
# 先建立 相關係數 的陣列
data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
data_corrs = data.corr()

# 繪製相關係數 (correlations) 的 Heatmap
plt.figure(figsize = (8, 6))
sns.heatmap(data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')
```

### Gridplot 
是一個預設的繪圖方式， seaborn 幫你分析個變數間的關係，結合 scatter plot 與 historgram 的好處來呈現變數間的相關程度

#### example

```py
# 定義我們的繪圖樣本數
N_sample = 1000
# 把 NaN 數值刪去, 並限制資料上限為 1000 : 因為要畫點圖, 如果點太多，會畫很久!
plot_data = plot_data.dropna().sample(n = N_sample)
# 建立 pairgrid 物件
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

# 我們可以定義不同的畫圖方式
# 上半部為 scatter
grid.map_upper(plt.scatter, alpha = 0.2)

# 對角線畫 histogram
grid.map_diag(sns.kdeplot)

# 下半部放 density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05)
plt.show()
```

## Day 16 : 模型初體驗 Logistic Regression (Kaggle)

先體驗 kaggle 使用過程，模型的詳細說明之後講

![](https://i.imgur.com/VDZNZI6.jpg)

---

## Day 17 : 特徵工程簡介

![](https://i.imgur.com/nl40vRy.jpg)

特徵工程就是我們將一個事實(例如性別、房屋坪數、官階等)，轉換到對應分數的過程。

這個過程通常是在 **資料彙整之後** 及 **訓練模型之前**

我們在做特徵工程時，必須去設計中間的轉換過程，將 Fact 轉換成 Score，之後再拿去訓練模型

![](https://i.imgur.com/F2nKG1o.png)

參考連結 : [link](https://www.zhihu.com/question/29316149)

這邊先示範一個簡單的特徵工程
```py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()

# 針對 dtype 為 'object' 的欄位先做 LabelEncoder，轉為數值型態
# 再使用 MinMaxScaler 去做Normalization，降低不同特徵間的影響
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()
```

## Day 18 : 特徵類型

### 常見特徵類型
* 數值型特徵 (int or float)
* 類別型特徵 (object : 如型號、班級等)
* 二元型特徵 (True or False)
* 排序型特徵 (百分比等級、排名)
* 時間型特徵 (年月日、時分秒)

### 不同類型的特徵工程
* 數值型特徵 --> 可能用函式或條件式轉換分數
* 類別型特徵 --> 通常每一種類別對應道不同分數
* 二元型特徵 --> 通常是轉換成 1(True) 和 0(False)
* 排序型特徵 --> 有大小關係、非連續數字，通常直接轉成數值形態處理
* 時間型特徵 --> 最難處理的，不能轉為數值或類別，會失去它的週期性及排序性(在 Day 25 說明)

## Day 19 : 數值型特徵 - 補缺失值與標準化

### 填補缺值要怎麼補 ?
填補缺值最重要的是**欄位的領域知識**與**欄位中的非缺數值**

![](https://i.imgur.com/WXcRUiz.png)

#### 填補統計值
* 類別型欄位 : 填補眾數
* 數值型欄位 : 填補 Mean(不影響群體的偏態)、填補 Median(較影響群體的偏態)

#### 填補指定值
* 補 0 : 空缺原本就有 0 的含意 (如房間數量沒數值，就補0)
* 補不可能出現的值 : 類別型欄位，當眾數不適合十，用特殊的值代表空缺值

#### 填補預測值
* 較為精確，但速度慢，需要有 Data Science 的概念
* 若填補範圍廣，且是重要特徵欄位時可用本方式，但須提防 overfitting

#### 補充資料
[如何填補缺失值](https://juejin.im/post/5b5c4e6c6fb9a04f90791e0c)

### 為何要標準化 ?

舉一個簡單的例子，四個評審給 0~10 分的範圍，每位評審的給分標準不一樣，這樣就不太公平了
![](https://i.imgur.com/TnbIxPI.png)

![](https://i.imgur.com/DrFHBgS.png)

常用的標準化方法有兩種
* 標準化 (Standard Scaler) : 適用於數值為常態分佈
* 最大最小化 (MinMax Scaler) : 適用於數值為均勻分佈

![](https://i.imgur.com/ArWeMDD.png)

#### 標準化/最小最大化的適用場合
* 樹狀狀模型或非樹狀狀模型
  * 非樹狀模型 (如線性迴歸, 羅吉斯迴歸, 類神經...等) : 會有影響
  * 樹狀模型 (如決策樹, 隨機森林林, 梯度提升樹...等) : 不會有影響
* 標準化 / 最小最大化使用上的差異
  * 標準化 : 轉換不易受到極端值影響
  * 最小最大化 : 轉換容易受到極端值影響

### 總結
* 補缺失值的方法因特徵類型與缺的意義不同，會有許多不同補法，需要因資料調整，無法一概而論
* 補缺失值還要注意盡量**不要破壞資料分布**
* 標準化的意義：平衡數值特徵間的影響力
* 因為最大最小化對極端數值較敏感，所以如果資料不會有極端值，或已經去極端值，就適合用最大最小化，否則請用標準化

### code
```python
# 查看哪些欄位是 null，並把它排在前面
df.isnull().sum().sort_values(ascending=False).head()

# 把數值(int、float)欄位取出來
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 拿去訓練模型(linear regression)，空值去補平均值
df_mn = df.fillna(df.mean())
train_X = df_mn[:train_num]
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
```

## Day 20 : 數值型特徵 - 去除離群值
如果只有少數幾筆資料跟其他數值差異很大，標準化無法處理

**處理離群值的方法有兩種**
* 把它去除掉
* 把它做調整
  * 用 ln 去轉換
  * 縮尾或截尾 (取 第2.5百分位 至 第97.5百分位 為範圍) 

除去離群值，將可使得模型預測較為準確，但隨意刪除可能也會刪掉重要訊息

### code
```python
# 調整離群值，df[column].clip(x,y)
# .clip 可以將超過這範圍的值都轉成範圍內
# ex : 小於800的變成800，大於2500變成2500
df['column_to_change'] = df['column_to_change'].clip(800, 2500)

train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

# 刪除離群值
keep_index = (df['column_to_change']>800) & (df['column_to_change']<2500)
df = df[keep_index]
train_Y = train_Y[keep_index]

train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
```

## Day 21 : 數值型特徵-去除偏態
**在哪些情況下，需要對資料進行去偏態 ?**
* 離群資料比例太高時
* 平均值沒有代表性時

**去除偏態的方法有哪幾種**
* 對數去偏 (log1p : 常用於 **計數/價格** 這類非負且可能為0的欄位，因為可能為0，所以先加1，再取對數)
* 方根去偏 (將數值減去最小值後，再開根號，常用於**最大值有限時適用**)
* 分布去偏 (使用 boxcox 轉換函數來調整，可以調整參數來看分布，但被轉換的值記得要為正，不可為負)

![](https://i.imgur.com/nFy8DMn.png)

boxcox 的 lambda 參數表
* lambda = 0, 取 log
* lambda = 0.5, 方根去偏

### code
Data
```python
# 對數去偏
df_fixed['column_to_fix'] = np.log1p(df_fixed['column_to_fix'])

# 方根去偏
from scipy import stats
df_fixed['LotArea'] = stats.boxcox(df_fixed['LotArea'], lmbda=0.5)

# 分布去偏(可以調 lmbda 參數 : 0~ 0.5 間)
df_fixed['LotArea'] = stats.boxcox(df_fixed['LotArea'], lmbda=0.15)
```
## Day 22 : 類別型特徵 - 基礎處理

基本的處理方式有兩類 
* **Label Encoding** : 
  * 依出現的次序依序給它編碼，流水號的概念
  * 缺點是數字的大小及次序，並無實質上的意義
  * 適用於 **樹狀模型**
* **One Hot Encoding** : 
  * 為了改良數字大小的影響，將每一種類別都獨立成一個 column
  * 記憶體佔用空間大，增加計算時間，類別越多越嚴重
  * 適用於 **非樹狀模型**

建議採取方式
* 預先都先使用 Label Encoding
* 除非該特徵重要性高，且類別數量較少，則可考慮 one hot encoding

### code
Label Encoding 
```python
# Label Encoder 轉換器
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

df_temp = pd.DataFrame()
for c in df.columns:
    # 每個 object 欄位都經過 Label encoder 轉換
    df_temp[c] = LabelEncoder().fit_transform(df[c])

train_X = df_temp[:train_num]
estimator = LinearRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')
```


One Hot Encoding
```python 
# pandas.get_dummies 為內建的 one_hot_encoding 轉換器
from sklearn.linear_model import LinearRegression

df_temp = pd.get_dummies(df)

train_X = df_temp[:train_num]
estimator = LinearRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')
```

## Day 23 : 類別型特徵 - 均值編碼

**如果類別特徵看起來與目標值有顯著相關，應該如何編碼?**

ex : 所居地 (類別特徵)  v.s. 房價 (目標值)

這時候我們在將 **類別特徵** 做特徵工程時，可以將 **目標值的平均** 作為轉換的數值

![](https://i.imgur.com/M5W4IPh.png)

但當我們的樣本數非常少 ， 且樣本中剛好又有極端值時 ， 若只單純做平均會對我們的數據造成極大的誤差

![](https://i.imgur.com/86qatWo.png)

因此我們必須考慮樣本數，當作可靠度的參考
* 樣本數多 ， 可靠度高 : 相信類別平均
* 樣本數少 ， 可靠度低 : 相信所有類別總平均的結果

為了避免均質化對於數據有影響，通常會進行 **均值平滑化**

![](https://i.imgur.com/KcdB43q.png)

```python
# 只留類別型欄位
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
df = df[object_features]

# 使用 groupby 協助做均質化
for c in df.columns:
    mean_df = df.groupby([c])['Target_col'].mean().reset_index()
    mean_df.columns = [c, f'{c}_mean']
    data = pd.merge(data, mean_df, how='left')
    # 做完均質化記得把原先的去掉
    data = data.drop([c] , axis=1)
```

## Day 24 : 類別型特徵 - 其他進階處理

### 計數編碼 (Counting)

如果 **類別的目標均價** 與 **類別比數** 呈正相關(或負相關)，我們也可以考慮將 **筆數** 當作我們的特徵之一

![](https://i.imgur.com/PAmgUbe.png)

像是在自然語言處理(NLP)，字詞的計數可以當作 詞頻，是重要特徵之一

### 特徵雜湊 ( Feature Hash )

類別型特徵最麻煩的問題 : 相異類別的數量非常龐大, 該如何編碼?

## Day 25 : 時間型特徵

時間型特徵的格式多為 **2014-06-12 03:25:42** 這一類，那我們要如何把它轉換成欄未來處理呢?

這邊提供以下 3 種方法來使用

### 時間型特徵分解

這是最為直接的處理方式，直接將這特徵分解成 **月、日、年、時、分、秒** 這幾個欄位來處理，但是有些欄位又和我們想分析的結果關聯性較小，這並非是最好的分類方式

### 週期循環特徵

時間也有週期的概念念, 可以用週期合成⼀些重要的特徵
* 年週期 ： 春下秋冬氣溫等
* 月週期 ： 薪水、繳費
* 周週期 ： 週休、消費習慣
* 日週期 ： 睡眠時間

因為是週期資料，所以到一個循環還是要首尾相連，可使用 **cos** 及 **sin** 來做運算

![](https://i.imgur.com/210Wf0c.png)


### 時段特徵 

短暫時段內的事件計數，也可能影響事件發生的機率，例如網站觀看時間，
，這時我們可以取**時段**作為我們的新欄位
