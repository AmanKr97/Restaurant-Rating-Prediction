
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('src/zomato_data.csv')

print(df.head())
x= df.drop('rate', axis=1)
y =df['rate']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)

from sklearn.ensemble import ExtraTreesRegressor
et = ExtraTreesRegressor()
et.fit(x_train,y_train)

et_pred=et.predict(x_test)


import pickle
pickle.dump(et,open('src/model.pkl','wb'))
model = pickle.load(open('src/model.pkl','rb'))
print(et_pred)