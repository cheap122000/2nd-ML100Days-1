# 各主題精華重點說明(待更新)

* **1-資料清理數據前處理**
* **2-資料科學特徵工程技術**
* **3-機器學習基礎模型建立**
* **4-機器學習調整參數**
* **期中考-Kaggle競賽**
* **5-非監督式機器學習**
* **6-深度學習理論與實作**
* **7-初探深度學習使用Keras**
* **8-深度學習應用卷積神經網路**
* **期末考-Kaggle競賽**
* **結語**

### :point_right: 我的[Kaggle](https://www.kaggle.com/kuoyuhong)

# 主題一：資料清理數據前處理

![前處理](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E5%89%8D%E8%99%95%E7%90%86.png)
![探索式數據分析](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E6%8E%A2%E7%B4%A2%E5%BC%8F%E6%95%B8%E6%93%9A%E5%88%86%E6%9E%90.png)

### **重點摘要：**
### **Day_005_HW** － EDA資料分佈：
### **Day_006_HW** － EDA: Outlier 及處理：
### **Day_008_HW** － DataFrame operationData frame merge/常用的 DataFrame 操作：
### **Day_011_HW** － EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)：
### **Day_014_HW** － Subplots：
### **Day_015_HW** － Heatmap & Grid-plot：
### **Day_016_HW** － 模型初體驗 Logistic Regression：

## 一、Outlier處理、資料標準化、離散化：

### 檢查異常值(Outlier)的方法：
**統計值**：如平均數、標準差、中位數、分位數、z-score、IQR<br>
**畫圖**：如直方圖、盒圖、次數累積分布等<br>
**處理異常值**：
* 取代補值：中位數、平均數等
* 另建欄位
* 整欄不用

#### z-score：<br>
Z = ( x - np.mean(x) ) / np.std(x)

sklearn有內建的z-score方法可以使用<br>
```
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
train_set_scaled = pd.DataFrame(scale.fit_transform(train_set), columns=train_set.keys())
```

#### IQR(四分位數間距)：<br>
np.quantile(item, 0.5)<br>

Q1 = item.quantile(0.25)<br>
Q3 = item.quantile(0.75)<br>
IQR = Q3 - Q1<br>

### 資料標準化：
Python標準化預處理函數：<br>

```
preprocessing.scale(X,axis=0, with_mean=True, with_std=True, copy=True)
```
將數據轉化為標準常態分佈(均值為0，方差為1)<br>

```
preprocessing.MinMaxScaler(X,feature_range=(0, 1), axis=0, copy=True)
```
將數據在縮放在固定區間，默認縮放到區間[0, 1]<br>
min_：ndarray，縮放後的最小值偏移量<br>
scale_：ndarray，縮放比例<br>
data_min_：ndarray，資料最小值<br>
data_max_：ndarray，資料最大值<br>
data_range_：ndarray，資料最大最小範圍的長度<br>

```
preprocessing.maxabs_scale(X,axis=0, copy=True)
```
數據的縮放比例為絕對值最大值，並保留正負號，即在區間[-1.0, 1.0]內。唯一可用於稀疏數據scipy.sparse的標準化<br>
scale_：ndarray，縮放比例<br>
max_abs_：ndarray，絕對值最大值<br>
n_samples_seen_：int，已處理的樣本個數<br>

```
preprocessing.robust_scale(X,axis=0, with_centering=True, with_scaling=True,copy=True)
```
通過Interquartile Range (IQR) 標準化數據，即四分之一和四分之三分位點之間<br>

```
preprocessing.StandardScaler(copy=True, with_mean=True,with_std=True)
```
scale_：ndarray，縮放比例<br>
mean_：ndarray，均值<br>
var_：ndarray，方差<br>
n_samples_seen_：int，已處理的樣本個數，調用partial_fit()時會累加，調用fit()會重設<br>

#### 標準化方法：<br>
* fit(X[,y])：根據數據X的值，設置標準化縮放的比例
* transform(X[,y, copy])：用之前設置的比例標準化X
* fit_transform(X[, y])：根據X設置標準化縮放比例並標準化
* partial_fit(X[,y])：累加性的計算縮放比例
* inverse_transform(X[,copy])：將標準化後的數據轉換成原數據比例
* get_params([deep])：獲取參數
* set_params( **params)：設置參數

### 資料離散化：
為什麼要離散化？<br>
* 調高計算效率
* 分類模型計算需要
* 給予距離計算模型（k均值、協同過濾）中降低異常資料對模型的影響
* 影象處理中的二值化處理

#### 連續資料離散化方法：<br>
* 分位數法：使用四分位、五分位、十分位等進行離散
* 距離區間法：等距區間或自定義區間進行離散，有點是靈活，保持原有資料分佈
* 頻率區間法：根據資料的頻率分佈進行排序，然後按照頻率進行離散，好處是資料變為均勻分佈，但是會更改原有的資料結構
* 聚類法：使用k-means將樣本進行離散處理
* 卡方：通過使用基於卡方的離散方法，找出資料的最佳臨近區間併合並，形成較大的區間
* 二值化：資料跟閾值比較，大於閾值設定為某一固定值（例如1），小於設定為另一值（例如0），然後得到一個只擁有兩個值域的二值化資料集。

```
pd.cut(item, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, labels=NULL)
```
x ： 必須是一維資料<br>
bins： 不同面元（不同範圍）型別:整數，序列如陣列, 和IntervalIndex<br>
right： 最後一個bins是否包含最右邊的資料，預設為True<br>
precision：精度 預設保留三位小數<br>
retbins： 即return bins 是否返回每一個bins的範圍 預設為False<br>
labels(表示結果標籤，一般最好新增，方便閱讀和後續統計)<br>

```
pandas.qcut(x, q, labels=None, retbins=False, precision=3, duplicates=’raise’)
```
x:是資料 1d ndarray或Series<br>
q：整數或分位數陣列；定義區間分割方法<br>
分位數10為十分位數，4為四分位數等。或分位陣列，如四分位數 [0, 0.25, 0.5, 0.75, 1] 分成兩半[0, 0.5, 1]<br>

![連續型數值標準化](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E9%80%A3%E7%BA%8C%E5%9E%8B%E6%95%B8%E5%80%BC%E6%A8%99%E6%BA%96%E5%8C%96.png)

---

## 二、資料視覺化(Matplotlib、Seaborn)：

### matplotlib方法：
import matplotlib.pyplot as plt<br>

#### plt.plot：
plt.figure() #定義一個圖像視窗<br>
plt.title('標題')<br>
plt.xlabel('X軸名稱')<br>
plt.ylabel('Y軸名稱')<br>
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--') #虛線<br>
linewidth：曲線寬度<br>
linestyle：曲線類型<br>
plt.xlim((-1, 2)) #x座標範圍<br>
plt.ylim((-2, 3)) #y座標範圍<br>
plt.xlabel('I am x') #x座標軸名稱<br>
plt.ylabel('I am y') #y座標軸名稱<br>
plt.xticks([坐標刻度],[標籤])<br>
plt.yticks([0,1,2,3,4],['$A$','$B$','C','D','E']) #設置x,y坐標軸刻度及標籤，$$是設置字體<br>
ax = plt.gca() #獲取當前的坐標軸，gca = (get current axis)的縮寫<br>
plot.kde() 創建一個核密度的繪圖，對於 Series和DataFrame資料結構都適用<br>
label = 'target == 1'：在圖表中顯示說明的圖例<br>
![plt.plot](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/plt.plot.png)

#### plt.hist()：
range = (0, 100000) #X軸最大最小值定義<br>
color = 'skyblue' #顏色設定<br>
set_title('標題')<br>
set_xlabel('X軸名稱')<br>
set_ylabel('Y軸名稱')<br>
arr: 需要計算直方圖的一維數組<br>
bins: 直方圖的柱數，默認為10<br>
density: : 是否將得到的直方圖向量歸一化。默認為0<br>
color：顏色序列，默認為None<br>
facecolor: 直方圖顏色；<br>
edgecolor: 直方圖邊框顏色<br>
alpha: 透明度<br>
histtype: 直方圖類型，『bar』, 『barstacked』, 『step』, 『stepfilled』：<br>
histtype='xxxx' 設定長條圖的格式: bar與stepfilled爲不同形式的長條圖, step以橫線標示數值.
* 'bar'是傳統的條形直方圖。如果給出多個數據，則條並排排列。
* 'barstacked'是一種條形直方圖，其中多個數據堆疊在一起。
* 'step'生成一個默認未填充的線圖。
* 'stepfilled'生成一個默認填充的線圖。

normed : boolean, optional， 意義就是說，返回的第一個n（後面解釋它的意義）吧，把它們正則化它，讓bins的值 的和為1，這樣差不多相當於概率分佈似的了；<br>
cumulative : boolean, optional ，每一列都把之前的加起來。<br>
bottom : array_like, scalar, or None，下面的每個bin的基線，表示bin的值都從這個基線上往上加；<br>
orientation : {‘horizontal’, ‘vertical’}, optional：指的方向，分為水準與垂直兩個方向。
rwidth : scalar or None, optional ，控制你要畫的bar 的寬度；<br>
![plt.hist](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/plt.hist.png)

#### plt.boxplot：
matplotlib包中boxplot函數的參數含義及使用方法：<br>
plt.boxplot(x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None)<br>
x：指定要繪製箱線圖的數據；<br>
notch：是否是凹口的形式展現箱線圖，默認非凹口；<br>
sym：指定異常點的形狀，默認為+號顯示；<br>
vert：是否需要將箱線圖垂直擺放，默認垂直擺放；<br>
whis：指定上下須與上下四分位的距離，默認為1.5倍的四分位差；<br>
positions：指定箱線圖的位置，默認為[0,1,2…]；<br>
widths：指定箱線圖的寬度，默認為0.5；<br>
patch_artist：是否填充箱體的顏色；<br>
meanline：是否用線的形式表示均值，默認用點來表示；<br>
showmeans：是否顯示均值，默認不顯示；<br>
showcaps：是否顯示箱線圖頂端和末端的兩條線，默認顯示；<br>
showbox：是否顯示箱線圖的箱體，默認顯示；<br>
showfliers：是否顯示異常值，默認顯示；<br>
boxprops：設置箱體的屬性，如邊框色，填充色等；<br>
labels：為箱線圖添加標籤，類似於圖例的作用；<br>
filerprops：設置異常值的屬性，如異常點的形狀、大小、填充色等；<br>
medianprops：設置中位數的屬性，如線的類型、粗細等；<br>
meanprops：設置均值的屬性，如點的大小、顏色等；<br>
capprops：設置箱線圖頂端和末端線條的屬性，如顏色、粗細等；<br>
whiskerprops：設置須的屬性，如顏色、粗細、線的類型等；<br>
#默認patch_artist=False，所以我們需要指定其參數值為True，即可自動填充顏色<br>

plt.show() #在任何環境下都能夠產生圖像<br>

![plt.boxplot](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/plt.boxplot.png)

### Seaborn方法：
import seaborn as sns<br>

#### sns.heatmap：
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)<br>
vmin, vmax : 顯示的數據值的最大和最小的範圍<br>
cmap : matplotlib顏色表名稱或對象，或顏色列表，可選從數據值到色彩空間的映射。如果沒有提供，默認設置<br>
center :指定色彩的中心值<br>
robust : 如果“Ture”和“ vmin或” vmax不存在，則使用強分位數計算顏色映射範圍，而不是極值<br>
annot :如果為True，則將數據值寫入每個單元格中<br>
fmt :表格里顯示數據的類型<br>
linewidths :劃分每個單元格的線的寬度。<br>
linecolor :劃分每個單元格的線的顏色<br>
cbar :是否繪製顏色條：colorbar，默認繪製<br>
cbar_kws :未知 cbar_ax :顯示xy坐標，而不是節點的編號<br>
square :為'True'時，整個網格為一個正方形<br>
xticklabels, yticklabels :可以以字符串進行命名，也可以調節編號的間隔，也可以不顯示坐標<br>
mask：布爾數組或DataFrame，可選，如果傳遞，則數據不會顯示在mask為True的單元格中。具有缺失值的單元格將自動被屏蔽。<br>
ax： matplotlib Axes，可選，用於繪製圖的軸，否則使用當前活動的Axes。<br>
kwargs：其他關鍵字參數，所有其他關鍵字參數都傳遞給ax.pcolormesh。<br>
![seaborn.heatmap](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/seaborn.heatmap.png)

### 查看總共有哪些畫圖樣式：
```
print(plt.style.available)<br>
print(type(plt.style.available))<br>
print(len(plt.style.available))<br>
['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']<br>
<class 'list'><br>
len = 26<br>
```
**使用plt.style.use('樣式')來套用方法**

![plt.style.available](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/plt.style.available.png)

### [Style sheets reference](https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html?highlight=pyplot%20text)

---

# 主題二：資料科學特徵工程技術

![特徵工程](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E7%89%B9%E5%BE%B5%E5%B7%A5%E7%A8%8B.png)

### **重點摘要：**
### **Day_019_HW** － 數值型特徵-補缺失值與標準化：
### **Day_020_HW** － 數值型特徵 - 去除離群值：
### **Day_022_HW** － 類別型特徵 - 基礎處理：
### **Day_023_HW** － 類別型特徵 - 均值編碼：
### **Day_024_HW** － 類別型特徵 - 其他進階處理：
### **Day_025_HW** － 時間型特徵：
### **Day_026_HW** － 特徵組合 - 數值與數值組合：
### **Day_027_HW** － 特徵組合 - 類別與數值組合：
### **Day_028_HW** － 特徵選擇：
### **Day_029_HW** － 特徵評估：
### **Day_030_HW** － 分類型特徵優化 - 葉編碼：

## 各類型特徵處理：

### 標籤編碼(Label Encoder)：
* 類似於流水號，依序將新出現的類別依序編上新代碼，已出現的類別編上已使用的代碼<br>
* 確實能轉成分數，但缺點是分數的大小順序沒有意義<br>
![標籤編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E6%A8%99%E7%B1%A4%E7%B7%A8%E7%A2%BC.png)

### 獨熱編碼(One Hot Encoder)：
* 為了改良數字大小沒有意義的問題，將不同的類別分別獨立為一欄<br>
* 缺點是需要較大的記憶空間與計算時間，且類別數量越多時越嚴重<br>
![獨熱編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E7%8D%A8%E7%86%B1%E7%B7%A8%E7%A2%BC.png)
* 類別型特徵建議預設採用標籤編碼，除非該特徵重要性高，且可能值較少(獨熱編碼時負擔較低) 時，才應考慮使用獨熱編碼<br>
* 獨熱編碼缺點是需要較大的記憶空間與計算時間，且類別數量越多時越嚴重<br>

類別型特徵有標籤編碼 (Label Encoding) 與獨熱編碼(One Hot Encoding) 兩種基礎編碼方式<br>
* 兩種編碼中標籤編碼比較常用<br>
* 當特徵重要性高，且可能值較少時，才應該考慮獨熱編碼<br>

### 均值編碼(Mean Encoding)：
* 使用時機：類別特徵看起來來與目標值有顯著相關時，使用目標值的平均值，取代原本的類別型特徵<br>
![均值編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E5%9D%87%E5%80%BC%E7%B7%A8%E7%A2%BC.png)
#### 平滑化：<br>
如果交易樣本非常少, 且剛好抽到極端值, 平均結果可能會有誤差很大<br>
因此, 均值編碼還需要考慮紀錄筆數, 當作可靠度的參考<br>
* 當平均值的可靠度低時, 我們會傾向相信全部的總平均<br>
* 當平均值的可靠度高時, 我們會傾向相信類別的平均<br>
* 依照紀錄筆數, 在這兩者間取折衷<br>

### 計數編碼(Counting)：
* 如果類別的目標均價與類別筆數呈正相關(或負相關)，也可以將筆數本身當成特徵例如 : 購物網站的消費金額預測<br>
![計數編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E8%A8%88%E6%95%B8%E7%B7%A8%E7%A2%BC.png)

#### 方法：
```
# 加上 'Ticket' 欄位的計數編碼
# 第一行 : df.groupby(['Ticket']) 會輸出 df 以 'Ticket' 群聚後的結果, 但因為群聚一類只會有一個值, 因此必須要定義運算
# 例如 df.groupby(['Ticket']).size(), 但欄位名稱會變成 size, 要取別名就需要用語法 df.groupby(['Ticket']).agg({'Ticket_Count':'size'})
# 這樣出來的計數欄位名稱會叫做 'Ticket_Count', 因為這樣群聚起來的 'Ticket' 是 index, 所以需要 reset_index() 轉成一欄
# 因此第一行的欄位, 在第三行按照 'Ticket_Count' 排序後, 最後的 DataFrame 輸出如 Out[5]
count_df = df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
# 但是上面資料表結果只是 'Ticket' 名稱對應的次數, 要做計數編碼還需要第二行 : 將上表結果與原表格 merge, 合併於 'Ticket' 欄位
# 使用 how='left' 是完全保留原資料表的所有 index 與順序
df = pd.merge(df, count_df, on=['Ticket'], how='left')
count_df.sort_values(by=['Ticket_Count'], ascending=False).head(10)
```

### 特徵雜湊(Feature Hash)：
使用時機：相異類別的數量量非常龐大時，特徵雜湊是一種折衷方案<br>
* 將類別由雜湊函數定應到一組數字，調整雜湊函數對應值的數量<br>
* 在計算空間/時間與鑑別度間取折衷<br>
* 也提高了訊息密度，減少無用的標籤<br>
![特徵雜湊](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E7%89%B9%E5%BE%B5%E9%9B%9C%E6%B9%8A.png)

#### 方法：
```
# 這邊的雜湊編碼, 是直接將 'Ticket' 的名稱放入雜湊函數的輸出數值, 為了要確定是緊密(dense)特徵, 因此除以10後看餘數
# 這邊的 10 是隨機選擇, 不一定要用 10, 同學可以自由選擇夠小的數字試看看. 基本上效果都不會太好
df_temp['Ticket_Hash'] = df['Ticket'].map(lambda x:hash(x) % 10)
```

* 計數編碼是計算類別在資料中的出現次數，當目標平均值與類別筆數呈正/負相關時，可以考慮使用<br>
* 當相異類別數量相當大時，其他編碼方式效果更更差，可以考慮雜湊編碼以節省時間<br>
*註 : 雜湊編碼效果也不佳，這類問題更好的解法是嵌入式編碼(Embedding)<br>

### 時間型特徵：

時間也有週期的概念, 可以用週期合成一些重要的特徵聯聯想看看 : 有哪幾種時間週期, 可串聯到一些可做特徵的性質?<br>
* 年週期與春夏秋冬季節溫度相關<br>
* 月週期與薪水、繳費相關<br>
* 周週期與周休、消費習慣相關<br>
* 日週期與生理理時鐘相關<br>

前述的週期所需數值都可由時間欄位組成, 但還首尾相接<br>
因此週期特徵還需以正弦函數( sin )或餘弦函數( cos )加以組合<br>
* 例如 : 
  * 年週期 ( 正 : 冷 / 負 : 熱 )cos((⽉月/6 + ⽇日/180 )π)<br>
  * 周週期 ( 正 : 精神飽滿/ 負 : 疲倦 )sin((星期幾/3.5 + ⼩小時/84 )π)<br>
  * 日週期 ( 正 : 精神飽滿 / 負 : 疲倦 )sin((⼩小時/12 + 分/720 + 秒/43200 )π)<br>

* 時間型特徵最常用的是特徵分解 - 拆解成年/月/日/時/分/秒的分類值<br>
* 週期循環特徵是將時間"循環"特性改成特徵方式, 設計關鍵在於首尾相接, 因此我們需要使用 sin /cos 等週期函數轉換<br>
* 常見的週期循環特徵有 - 年週期(季節) / 周周期(例假日) / 日週期(日夜與生活作息), 要注意的是最高與最點的設置<br>

### 群聚編碼：
* 類似均值編碼的概念，可以取類別平均值 (Mean) 取代險種作為編碼<br>
* 但因為比較像性質描寫，因此還可以取其他統計值，如中位數 (Median)，眾數(Mode)，最大值(Max)，最小值(Min)，次數(Count)...等<br>
![群聚編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E7%BE%A4%E8%81%9A%E7%B7%A8%E7%A2%BC.png)
![均值編碼&群聚編碼比較](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E5%9D%87%E5%80%BC%E7%B7%A8%E7%A2%BC%26%E7%BE%A4%E8%81%9A%E7%B7%A8%E7%A2%BC%E6%AF%94%E8%BC%83.png)

### 葉編碼(leaf encoding)：
* 採用決策樹的葉點作為編碼依據重新編碼<br>
* 每棵樹視為一個新特徵，每個新特徵均為分類型特徵，決策樹的葉點與該特徵標籤一一對應<br>
* 最後再以邏輯斯迴歸合併預測<br>
![葉編碼-1](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E8%91%89%E7%B7%A8%E7%A2%BC-1.png)

##### 葉編碼(leaf encoding)+邏輯斯迴歸：
* 葉編碼需要先對樹狀模型擬合後才能生成，如果這步驟挑選了較佳的參數，後續處理效果也會較好，這點與特徵重要性類似<br>
* 實際結果也證明，在分類預測中使用樹狀模型，再對這些擬合完的樹狀模型進行葉編碼+邏輯斯迴歸，通常會將預測效果再進一步提升<br>
![葉編碼-2](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E8%91%89%E7%B7%A8%E7%A2%BC-2.png)

* 葉編碼的目的是重新標記資料，以擬合後的樹狀模型分歧條件，將資料離散化，這樣比人為寫作的判斷條件更精準，更符合資料的分布情形<br>
* 葉編碼編完後，因為特徵數量較多，通常搭配邏輯斯迴歸或者分解機做預測，其他模型較不適合<br>

### 機器學習中的優化循環：

* 機器學習特徵優化，循環方式如圖<br>
![機器學習中的優化循環](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E4%B8%AD%E7%9A%84%E5%84%AA%E5%8C%96%E5%BE%AA%E7%92%B0.png)
* 其中增刪特徵指的是<br>
  * 特徵選擇(刪除)<br>
    * 挑選門檻，刪除一部分特徵重要性較低的特徵<br>
  * 特徵組合(增加)<br>
    * 依領域知識，對前幾名特徵做特徵組合或群聚編碼，形成更強力特徵<br>
* 由交叉驗證確認特徵是否有改善，若沒有改善則回到上一輪重選特徵增刪<br>
* 這樣的流程圖綜合了PART 2 : 特徵工程的主要內容，是這個部分的核心知識<br>

### 排列重要性(permutation Importance)：
* 雖然特徵重要性相當實用，然而計算原理必須基於樹狀模型，於是有了可延伸至非樹狀模型的排序重要性<br>
* 排序重要性計算，是打散單一特徵的資料排序順序，再用原本模型重新預測，觀察打散前後誤差會變化多少<br>
![排列重要性](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E6%8E%92%E5%88%97%E9%87%8D%E8%A6%81%E6%80%A7.png)

---

# 主題三：機器學習基礎模型建立

![模型選擇](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E6%A8%A1%E5%9E%8B%E9%81%B8%E6%93%87.png)

### **重點摘要：**

### **Day_034_HW** － 訓練/測試集切分的概念：
### **Day_035_HW** － regression vs. classification：
### **Day_036_HW** － 評估指標選定/evaluation metrics：
### **Day_037_HW** － regression model 介紹 - 線性迴歸/羅吉斯回歸：
### **Day_039_HW** － regression model 介紹 - LASSO 回歸/ Ridge 回歸：
### **Day_041_HW** － tree based model - 決策樹 (Decision Tree) 模型介紹：
### **Day_043_HW** － tree based model - 隨機森林 (Random Forest) 介紹：
### **Day_045_HW** － tree based model - 梯度提升機 (Gradient Boosting Machine) 介紹：

## 一、建立模型四步驟：

在 Scikit-learn 中，建立一個機器學習的模型其實非常簡單，流程大略是以下四個步驟<br>

1. 讀進資料，並檢查資料的 shape (有多少 samples (rows), 多少 features (columns)，label 的型態是什麼？)
    - 讀取資料的方法：
        * **使用 pandas 讀取 .csv 檔**：pd.read_csv
        * **使用 numpy 讀取 .txt 檔**：np.loadtxt 
        * **使用 Scikit-learn 內建的資料集**：sklearn.datasets.load_xxx
    - **檢查資料數量**：data.shape (data should be np.array or dataframe)
2. 將資料切為訓練 (train) / 測試 (test)
    - train_test_split(data)
3. 建立模型，將資料 fit 進模型開始訓練
    - clf = DecisionTreeClassifier()
    - clf.fit(x_train, y_train)
4. 將測試資料 (features) 放進訓練好的模型中，得到 prediction，與測試資料的 label (y_test) 做評估
    - clf.predict(x_test)
    - accuracy_score(y_test, y_pred)
    - f1_score(y_test, y_pred)

## 二、模型評估/模型驗證：

### 模型評估：
#### 評估指標-迴歸：
```
X, y = datasets.make_regression(n_features=1, random_state=42, noise=4) # 生成資料
model = LinearRegression() # 建立回歸模型
model.fit(X, y) # 將資料放進模型訓練
prediction = model.predict(X) # 進行預測
mae = metrics.mean_absolute_error(prediction, y) # 使用 MAE 評估
mse = metrics.mean_squared_error(prediction, y) # 使用 MSE 評估
r2 = metrics.r2_score(prediction, y) # 使用 r-square 評估
print("MAE: ", mae)
print("MSE: ", mse)
print("R-square: ", r2)
```

#### 評估指標-分類：<br>
##### AUC(Area Under Curve)：<br>
AUC 指摽是分類問題常用的指標，通常分類問題都需要定一個閾值(threshold) 來決定分類的類別 (通常為機率 > 0.5 判定為 1,  機率 < 0.5 判定為 0)<br>
AUC 是衡量曲線下的面積，因此可考量所有閾值下的準確性<br>
```
auc = metrics.roc_auc_score(y_test, y_pred) # 使用 roc_auc_score 來評估。這邊特別注意 y_pred 必須要放機率值進去!
print("AUC: ", auc) # 得到結果約 0.5，與亂猜的結果相近，因為我們的預測值是用隨機生成的
```
![auc for roc curves](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/auc%20for%20roc%20curves.png)

##### F1-Score：
分類問題中，我們有時會對某一類別的準確率特別有興趣。例如瑕疵/正常樣本分類，我們希望任何瑕疵樣本都不能被漏掉。<br>
Precision，Recall 則是針對某類別進行評估<br>
Precision: 模型判定瑕疵，樣本確實為瑕疵的比例<br>
Recall: 模型判定的瑕疵，佔樣本所有瑕疵的比例
(以瑕疵檢測為例例，若為 recall=1 則代表所有瑕疵都被找到)<br>
F1-Score 則是 Precision, Recall 的調和平均數<br>
```
threshold = 0.5
y_pred_binarized = np.where(y_pred>threshold, 1, 0) # 使用 np.where 函數, 將 y_pred > 0.5 的值變為 1，小於 0.5 的為 0
f1 = metrics.f1_score(y_test, y_pred_binarized) # 使用 F1-Score 評估
precision = metrics.precision_score(y_test, y_pred_binarized) # 使用 Precision 評估
recall  = metrics.recall_score(y_test, y_pred_binarized) # 使用 recall 評估
print("F1-Score: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)
```

### 模型驗證：
#### Model基礎驗證法：
```
from sklearn.datasets import load_iris # iris資料集
from sklearn.model_selection import train_test_split # 分割資料模組
from sklearn.neighbors import KNeighborsClassifier # K最近鄰(kNN，k-NearestNeighbor)分類演算法
#載入iris資料集
iris = load_iris()
X = iris.data
y = iris.target
#分割數據並
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
#建立模型
knn = KNeighborsClassifier()
#訓練模型
knn.fit(X_train, y_train)
#將準確率列印出
print(knn.score(X_test, y_test))
0.973684210526
#可以看到基礎驗證的準確率為0.973684210526
```

#### Model交叉驗證法(Cross Validation)：
```
from sklearn.cross_validation import cross_val_score # K折交叉驗證模組
#使用K折交叉驗證模組
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
#將5次的預測準確率列印出
print(scores)
[ 0.96666667  1.          0.93333333  0.96666667  1.        ]
#將5次的預測準確平均率列印出
print(scores.mean())
0.973333333333
#可以看到交叉驗證的準確平均率為0.973333333333
```

#### 以準確率(accuracy)判斷：
一般來說準確率(accuracy)會用於判斷分類(Classification)模型的好壞<br>
```
import matplotlib.pyplot as plt #視覺化模組
#建立測試參數集
k_range = range(1, 31)
k_scores = []
#藉由反覆運算的方式來計算不同參數對模型的影響，並返回交叉驗證後的平均準確率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
#視覺化數據
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
```

#### 以平均方差(Mean squared error)：
一般來說平均方差(Mean squared error)會用於判斷回歸(Regression)模型的好壞<br>
```
import matplotlib.pyplot as plt
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')
    k_scores.append(loss.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated MSE')
plt.show()
```

## 三、機器學習模型

### 線性回歸模型：

線性回歸模型 Linear Regression：<br>
* 簡單常見的線性模型，可使用於回歸問題。訓練速度非常快，但須注意資料共線性、資料標準化等限制
* 通常可作為 baseline 模型作為參考點

羅吉斯回歸 Logistics Regression：<br>
* 雖然有回歸兩個字，但 Logsitics 是分類模型
* 將線性回歸的結果，加上Sigmoid 函數，將預測值限制在 0 ~ 1 之間，即為預測機率值。
```
# 讀取breast_cancer資料集
breast_cancer = datasets.load_breast_cancer()

# 為方便視覺化，我們只使用資料集中的 1 個 feature (column)
X = breast_cancer.data[:, np.newaxis, 2]
print("Data shape: ", X.shape) # 可以看見有 442 筆資料與我們取出的其中一個 feature

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(X, breast_cancer.target, test_size=0.1, random_state=4)

# 建立一個線性回歸模型
regr = linear_model.LinearRegression()

# 將訓練資料丟進去模型訓練
regr.fit(x_train, y_train)

# 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)
```

### LASSO、Ridge Regression應用：

機器學習模型的目標函數中有兩個非常重要的元素<br>
* 損失函數 (Loss function)
* 正則化 (Regularization)

損失函數衡量預測值與實際值的差異，讓模型能往正確的方向學習<br>
正則化則是避免模型變得過於複雜，造成過擬合 (Over-fitting)<br>

正則化函數是用來衡量模型的複雜度<br>
該怎麼衡量？有 L1 與 L2 兩種函數<br>
![正則化函數-L1、L2](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E6%AD%A3%E5%89%87%E5%8C%96%E5%87%BD%E6%95%B8-L1%E3%80%81L2.png)

這兩種都是希望模型的參數數值不要太大，原因是參數的數值變小，噪音對最終輸出的結果影響越小，提升模型的泛化能力，但也讓模型的擬合能力下降<br>

LASSO 為 Linear Regression 加上 L1 <br>
Ridge 為 Linear Regression 加上 L2<br>
其中有個超參數α可以調整正則化的強度<br>
簡單來說，LASSO 與 Ridge 就是回歸模型加上不同的正則化函數<br>

Lasso 使用的是 L1 regularization，這個正則化的特性會讓模型變得較為稀疏，除了了能做特徵選取外，也會讓模型變得更輕量，速度較快<br>

---

### 決策樹DecisionTreeClassifier、DecisionTreeRegressor模型的應用：

決策樹 (Decision Tree)：<br>
從訓練資料中找出規則，讓每一次決策能使訊息增益 (Information Gain) 最大化<br>
訊息增益越大代表切分後的兩群資料，群內相似程度越高<br>

訊息增益 (Information Gain)：<br>
決策樹模型會用 features 切分資料，該選用哪個 feature 來切分則是由訊息增益的大小決定的。希望切分後的資料相似程度很高，通常使用吉尼係數來衡量相似程度<br>
![訊息增益](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E8%A8%8A%E6%81%AF%E5%A2%9E%E7%9B%8A.png)

衡量資料相似: Gini vs. Entropy：<br>
該怎麼衡量量資料相似程度？通常使用吉尼係數 (gini-index) 或熵 (entropy) 來衡量<br>
![衡量資料相似 Gini vs. Entropy](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E8%A1%A1%EF%A5%BE%E8%B3%87%E6%96%99%E7%9B%B8%E4%BC%BC%20Gini%20vs.%20Entropy.png)

應用：
```
# 讀取breast_cancer資料集
breast_cancer = datasets.load_breast_cancer()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.25, random_state=4)

# 建立模型
clf = DecisionTreeClassifier(criterion = 'gini',max_depth = None,min_samples_split = 2,min_samples_leaf = 1,)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)
```

決策樹的超參數：<br>
Criterion: 衡量資料相似程度的 <br>
metricMax_depth: 樹能生長的最深限制<br>
Min_samples_split: 至少要多少樣本以上才進行切分<br>
Min_samples_lear: 最終的葉子 (節點) 上至少要有多少樣本<br>

---

### 隨機森林RandomForest方法：
```
# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)
clf = RandomForestClassifier(n_estimators=20, max_depth=4)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

# 觀看準確率：
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# 觀看feature_names：
print(iris.feature_names)

# 觀看Feature importance：
print("Feature importance: ", clf.feature_importances_)
```

隨機森林的模型超參數：<br>
同樣是樹的模型，所以像是 max_depth, min_samples_split 都與決策樹相同<br>
可決定要生成數的數量，越多越不容易過擬和，但是運算時間會變長<br>
```
fromsklearn.ensemble import RandomForestClassifier #集成模型

clf = RandomForestClassifier(
n_estimators=10, #決策樹的數量量
criterion="gini",
max_features="auto", #如何選取 features
max_depth=10,
min_samples_split=2,
min_samples_leaf=1
)
```

---

### 梯度提升機 (Gradient Boosting Machine)：

隨機森林使用的集成方法稱為 Bagging (Bootstrap aggregating)，用抽樣的資料與 features ⽣生成每一棵樹，最後再取平均<br>
Boosting 則是另一種集成方法，希望能夠由後面生成的樹，來來修正前面樹學不好的地方
要怎麼修正前面學錯的地方呢？計算 Gradient!<br>
每次生成樹都是要修正前面樹預測的錯誤，並乘上 learning rate 讓後面的樹能有更多學習的空間<br>
Random Forest 的每一棵樹皆是獨立的樹，前一棵樹的結果不會影響下一顆<br>
Gradient boosting 因為下一棵樹是為了修正前一棵樹的錯誤，因此每一棵樹皆有相關聯<br>

Bagging 與 Boosting 的差別：<br>
* Bagging 是透過抽樣 (sampling) 的方式來生成每一棵樹，樹與樹之間是獨立生成的
* Boosting 是透過序列 (additive)的方式來生成每一顆樹，每棵樹都會與前面的樹關聯，因為後面的樹要能夠修正

可決定要生成數的數量，越多越不容易過擬和，但是運算時間會變長<br>
```
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(
loss="deviance", #Loss 的選擇，若改為 exponential 則會變成Adaboosting 演算法，概念相同但實作稍微不同
learning_rate=0.1, #每棵樹對最終結果的影響，應與 n_estimators 成反比
n_estimators=100 #決策樹的數量量
)
```

Q：隨機森林與梯度提升機的特徵重要性結果不相同？<br>
A：決策樹計算特徵重要性的概念是，觀察某一特徵被用來切分的次數而定。假設有兩個一模一樣的特徵，在隨機森林中每棵樹皆為獨立，因此兩個特徵皆有可能被使用，最終統計出來的次數會被均分。在梯度提升機中，每棵樹皆有關連，因此模型僅會使用其中一個特徵，另一個相同特徵的重要性則會消失<br>

---

# 主題四：機器學習調整參數

![參數調整](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E5%8F%83%E6%95%B8%E8%AA%BF%E6%95%B4.png)
![集成](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E5%8F%83%E6%95%B8%E8%AA%BF%E6%95%B4-%E9%9B%86%E6%88%90.png)

### **重點摘要：**

### **Day_047_HW** － 超參數調整與優化：
### **Day_049_HW** 集成方法 : 混合泛化(Blending)：
### **Day_050_HW** 集成方法 : 堆疊泛化(Stacking)：

## 一、超參數調整：

之前接觸到的所有模型都有超參數需要設置：
* LASSO，Ridge: α的大小
* 決策樹：樹的深度、節點最小樣本數
* 隨機森林：樹的數量
這些超參數都會影響模型訓練的結果，建議先使用預設值，再慢慢進行調整<br>
超參數會影響結果，但提升的效果有限，資料清理與特徵工程才能最有效的提升準確率，調整參數只是一個加分的工具<br>

### 超參數調整方法：<br>
* 窮舉法 (Grid Search)：直接指定超參數的組合範圍，每一組參數都訓練完成，再根據驗證集 (validation) 的結果選擇最佳參數
* 隨機搜尋 (Random Search)：指定超參數的範圍，用均勻分布進行參數抽樣，用抽到的參數進行訓練，再根據驗證集的結果選擇最佳參數
* 隨機搜尋通常都能獲得更佳的結果

### 正確的超參數調整步驟：
若持續使用同一份驗證集 (validation) 來調參，可能讓模型的參數過於擬合該驗證集，正確的步驟是使用 Cross-validation 確保模型泛化性<br>
1. 先將資料切分為訓練/測試集，測試集保留不使用
2. 將剛切分好的訓練集，再使用Cross-validation 切分 K 份訓練/驗證集
3. 用 grid/random search 的超參數進行訓練與評估
4. 選出最佳的參數，用該參數與全部訓練集建模
5. 最後使用測試集評估結果

Q：超參數調整對最終結果影響很大嗎？<br>
A：超參數調整通常都是機器學習專案的最後步驟，因為這對於最終的結果影響不會太多，多半是近一步提升 3-5 % 的準確率，但是好的特徵工程與資料清理是能夠一口氣提升 10-20 ％的準確率！因此建議專案一開始時，不需要花太多時間進行超參數的調整<br>

```
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
wine = datasets.load_wine()
# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=42)
# 建立模型
clf = GradientBoostingRegressor(random_state=7)
# 先看看使用預設參數得到的結果
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(metrics.mean_squared_error(y_test, y_pred))
# 設定要訓練的超參數組合
n_estimators = [100, 200, 300]
max_depth = [1, 3, 5]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
## 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
grid_search = GridSearchCV(clf, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
# 開始搜尋最佳參數
grid_result = grid_search.fit(x_train, y_train)
# 預設會跑 3-fold cross-validadtion，總共 9 種參數組合，總共要 train 27 次模型
# 印出最佳結果與最佳參數
print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
```

## 二、集成：

集成是使用不同方式，結合多個/多種不同分類器，作為綜合預測的做法統稱<br>
將模型截長補短，也可說是機器學習裡的和議制 / 多數決<br>
其中又分為資料面的集成 : 如裝袋法(Bagging) / 提升法(Boosting)<br>
以及模型與特徵的集成 : 如混合泛化(Blending) / 堆疊泛化(Stacking)<br>

#### 資料面集成 : 裝袋法 ( Bagging )：
* 裝袋法顧名思義，是將資料放入袋中抽取，每回合結束後全部放回袋中重抽
* 再搭配弱分類器取平均/多數決結果，最有名的就是前面學過的隨機森林

#### 資料面集成 : 提升法 ( Boosting )：
* 提升法則是由之前模型的預測結果，去改變資料被抽到的權重或目標值
* 將錯判資料被抽中的機率放大，正確的縮小，就是自適應提升 (AdaBoost, Adaptive Boosting)
* 如果是依照估計誤差的殘差項調整新目標值，則就是梯度提升機 (Gradient Boosting Machine) 的作法，只是梯度提升機還加上用梯度來選擇決策樹分支

#### 資料集成 v.s. 模型與特徵集成
兩者雖然都稱為集成，其實適用範圍差異很大，通常不會一起提及<br>

##### 資料集成：
Bagging / Boosting<br>
使用不同訓練資料 + 同一種模型，多次估計的結果合成最終預測<br>

##### 模型與特徵集成：
Voting / Blending / Stacking<br>
使用同一資料 + 不同模型，合成出不同預測結果<br>

### 混合泛化 ( Blending )：
其實混合泛化非常單純，就是將不同模型的預測值加權合成，權重和為 1如果取預測的平均 or 一人一票多數決(每個模型權重相同)，則又稱為投票泛化(Voting)<br>
雖然單純，但因為最容易使用且有效，至今仍然是競賽中常見的作法<br>
![](圖片)

##### 容易使用：
* 不只在一般機器學習中有用，影像處理或自然語言處理等深度學習，也一樣可以使用
* 因為只要有預測值(Submit 檔案)就可以使用，許多跨國隊伍就是靠這個方式合作
* 另一方面也因為只要用預測值就能計算，在競賽中可以快速合成多種比例的答案，妥善消耗掉每一天剩餘的 Submit 次數

##### 效果顯著：
* Kaggle 競賽截止日前的 Kernel，有許多只是對其他人的輸出結果做Blending，但是因為分數較高，因此也有許多人樂於推薦與發表
* 在2015年前的大賽中，Blending 仍是主流，例如林軒田老師也曾在課程中提及 : 有競賽的送出結果，是上百個模型的 Blending
* 注意事項：
  * Blending 的前提是 : 個別單模效果都很好(有調參)並且模型差異大，單模要好尤其重要，如果單模效果差異太大，Blending 的效果提升就相當有限

##### 重點：
* 資料工程中的集成，包含了資料面的集成 - 裝袋法(Bagging) / 提升法(Boosting)，以及模型與特徵的集成 - 混合泛化(Blending) / 堆疊泛化(Stacking)
* 混合泛化提升預測力的原因是基於模型差異度大，在預測細節上能互補，因此預測模型只要各自調參優化過且原理不同，通常都能使用混合泛化集成

---

### 堆疊泛化(Stacking)：

Stacking 小歷史<br>
雖然堆疊泛化 (Stacking) 的論文早在 2012 年，就由 David H. Wolpert 發布[原始論文連結](http://www.machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf)<br>
但真正被廣泛應用於競賽上，是2014年底的 Kaggle 競賽開始<br>
由於 Kaggle 一直有前幾名於賽後發布做法的風氣，所以當有越來越多的前幾名使用 Stacking 後，這個技術就漸漸變得普及起來，甚至後來出現了加速混合與計算速度的StackNet<br>

#### 相對於 Blending 的改良：
* 不只將預測結果混合，而是使用預測結果當新特徵
* 更進一步的運用了資料輔助集成，但也使得 Stacking 複雜許多

#### Stacking 的設計挑戰 : 訓練測試的不可重複性：
Blending 與 Stacking 都是模型集成，但是模型預測結果怎麼使用，是關鍵差異
![Stacking 的設計挑戰 訓練測試的不可重複性](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/Stacking%20%E7%9A%84%E8%A8%AD%E8%A8%88%E6%8C%91%E6%88%B0%20%20%E8%A8%93%E7%B7%B4%E6%B8%AC%E8%A9%A6%E7%9A%84%E4%B8%8D%E5%8F%AF%E9%87%8D%E8%A4%87%E6%80%A7.png)

##### Blending 與 Stacking 的原理差異：
* Stacking 主要是把模型當作下一階的特徵編碼器來使用，但是待編碼資料與用來訓練編碼器的資料不可重複 (訓練測試的不可重複性)
* 若將訓練資料切成兩組 : 待編碼資料太少，下一層的資料筆數就會太少，訓練編碼器的資料太少，則編碼器的強度就會不夠，這樣的困境該如何解決呢?

#### Stacking 最終設計 : 巧妙的 K-Fold 拆分：
* Stacking 最終採取了下圖設計 : 將資料拆成 K 份 (圖中 K=5)，每 1/K 的資料要編碼時，使用其他的 K-1 組資料訓練模型/編碼器
* 這樣資料就沒有變少，K 夠大時編碼器的強韌性也夠，唯一的問題就是計算時間隨著 K 變大而變長，但 K 可以調整，且相對深度學習所需的時間來來說，這樣的時間長度也還算可接受
![Stacking 最終設計-巧妙的 K-Fold 拆分](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/Stacking%20%E6%9C%80%E7%B5%82%E8%A8%AD%E8%A8%88-%E5%B7%A7%E5%A6%99%E7%9A%84%20K-Fold%20%E6%8B%86%E5%88%86.png)

#### 自我遞迴的 Stacking ?
大家在看到 Stacking 時可能已經注意到了 : 既然 Stacking 是在原本特徵上，用模型造出新特徵，那麼我們自然會想到兩個問題 :<br>
* Q1 能不能新舊特徵一起用，再用模型去預測呢?
* Q2 新的特徵，能不能再搭配模型創特徵，第三層第四層...一直下去呢?

另外三個問題：
* Q1：能不能新舊特徵一起用，再用模型預測呢?
* A1：可以，這裡其實有個有趣的思考，也就是 : 這樣不就可以一直一直無限增加特徵下去? 這樣後面的特徵還有意義嗎?不會 Overfitting 嗎?...其實加太多次是會 Overfitting 的，必需謹慎切分 Fold 以及新增次數
* Q2：新的特徵，能不能再搭配模型創特徵，第三層第四層...一直下去呢?
* A2：可以，但是每多一層，模型會越複雜 : 因此泛化(又稱為魯棒性)會做得更好，精準度也會下降，所以除非第一層的單模調得很好，否則兩三層就不需要繼續往下了
* Q3：既然同層新特徵會 Overfitting，層數加深會增加泛化，兩者同時用是不是就能把缺點互相抵銷呢?
* A3：可以!!而且這正是 Stacking 最有趣的地方，但真正實踐時，程式複雜，運算時間又要再往上一個量級，之前曾有大神寫過 StackNet 實現這個想法，用JVM 加速運算，但實際上使用時調參困難，後繼使用的人就少了

#### 真實世界的 Stacking 使用心得：
* Q1：實際上寫 Stacking 有這麼困難嗎?
* A1：其實不難，就像 sklearn 幫我們寫好了許多機器學習模型，mlxtend 也已經幫我們寫好了Stacking 的模型，所以用就可以了
* Q2：Stacking 結果分數真的比較高嗎?
* A2：不一定，有時候單模更高，有時候 Blending 效果就不錯，視資料狀況而定
* Q3：Stacking 可以做參數調整嗎?
* A3：可以，請參考 mlxtrend 的調參範例，主要差異是參數名稱寫法稍有不同
* Q4：還有其他做 Stacking 時需要注意的事項嗎?
* A4：「分類問題」的 Stacking 要注意兩件事：記得加上 use_probas=True(輸出特徵才會是機率值)，以及輸出的總特徵數會是：模型數量*分類數量(回歸問題特徵數=模型數量)

#### 重點：
* 堆疊泛化因為將模型預測當作特徵時，要避免要編碼的資料與訓練編碼器的資料重疊，因此設計上看起來來相當複雜
* 堆疊泛化理理論上在堆疊層數上沒有限制，但如果第一層的單模不夠複雜，堆疊二三層後，改善幅度就有限了
* 混合泛化相對堆疊泛化來來說，優點在於使用容易，缺點在於無法更深入的利用資料更進一步混合模型

```
# 堆疊泛化套件 mlxtend, 需要先行安裝(使用 pip 安裝即可)在執行環境下
from mlxtend.regressor import StackingRegressor

# 因為 Stacking 需要以模型作為第一層的特徵來源, 因此在 StackingRegressor 中,
# 除了要設本身(第二層)的判定模型 - meta_regressor, 也必須填入第一層的單模作為編碼器 - regressors
# 這裡第二層模型(meta_regressor)的參數, 一樣也需要用 Grid/Random Search, 請參閱講義中的 mlxtrend 網頁
meta_estimator = GradientBoostingRegressor(tol=10, subsample=0.44, n_estimators=100, 
                                           max_features='log2', max_depth=4, learning_rate=0.1)
stacking = StackingRegressor(regressors=[linear, gdbt, rf], meta_regressor=meta_estimator)

# 堆疊泛化預測檔 : 分數會依每次執行略有出入, 但通常 Public Score(競賽中的提交分數) 會再比單模好一些
# 雖然 Public Score 有可能比 Blending 分數略差, 但是因為不用依賴仔細調整的權重參數, 競賽結束時的 Private Score, 通常會比 Blending 好
# (因為權重依賴於 Public 的分數表現), 這種在未知 / 未曝光資料的預測力提升, 就是我們所謂 "泛化能力比較好" 在競賽/專案中的含意
stacking.fit(train_X, train_Y)
stacking_pred = stacking.predict(test_X)
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(stacking_pred)})
sub.to_csv('house_stacking.csv', index=False)
```

---

# 期中考：


