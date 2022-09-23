import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.data.csv')
print(df.isna().sum())
df['SL']=df['SL'].fillna(df['SL'].mean())
df['SW']=df['SW'].fillna(df['SW'].mean())
df['PL']=df['PL'].fillna(df['PL'].mean())
df['PW']=df['PW'].fillna(df['PW'].mean())


from sklearn.preprocessing import LabelEncoder
label_en = LabelEncoder()
a=['Classification']
for i in np.arange(len(a)):
    df[a[i]]=label_en.fit_transform(df[a[i]])
    
x=df.drop(['Classification'],axis=1   )
y=df['Classification']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=42, test_size=0.2)

from sklearn.svm import SVC
sv = SVC(kernel='rbf',random_state=0)
model=sv.fit(x_train,y_train)
y_pred=sv.predict(x_test)
pickle.dump(sv, open('model.pkl', 'wb'))