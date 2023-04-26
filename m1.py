import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
os.chdir("archive")

print(os.listdir())

d_name_1 = str(input("Enter name of your dataset_1 : "))
df_1 = pd.read_csv(d_name_1)
d_name_2 = str(input("Enter name of your dataset_2 : "))
df_2 = pd.read_csv(d_name_2)
frame = [df_1,df_2]
df = pd.concat(frame)
print(df.info())
df = df.replace(np.nan,df.mean())
print(df.head())

sns.relplot(x='IRRADIATION',y='MODULE_TEMPERATURE',hue= 'AMBIENT_TEMPERATURE',data = df)
plt.show()

train = df.drop(['DATE_TIME','PLANT_ID','SOURCE_KEY'],axis=1)
test = df['TOTAL_YIELD']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
le = LinearRegression()

x_train,x_test,y_train,y_test = train_test_split(train,test,test_size=0.2,random_state=100)
le.fit(x_train,y_train)

pred = le.predict(x_test)

print(pred)

acc = le.score(x_test,y_test)
print(acc)