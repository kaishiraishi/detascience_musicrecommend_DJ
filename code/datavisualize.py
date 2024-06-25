import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('track_features.csv')


stats = df.describe()


print(stats)


fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
fig.subplots_adjust(hspace=0.5, wspace=0.3)


axes = axes.ravel() 
for idx, col in enumerate(df.columns[2:]):  
    axes[idx].hist(df[col], bins=20, color='blue', alpha=0.7)
    axes[idx].set_title(col)
    axes[idx].set_ylabel('Frequency')

plt.show()


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('track_features.csv')


stats = df.describe()
print(stats)


str_stats = df.astype('str').describe()
print(str_stats)


numerical_cols = df.select_dtypes(include=['float64', 'int64'])
numerical_cols.boxplot(figsize=(12, 8))
plt.xticks(rotation=45)  
plt.title('Boxplot of Numerical Features')
plt.ylabel('Value')
plt.grid(True)
plt.show()
