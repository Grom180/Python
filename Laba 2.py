import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic (1).csv')
print(df.info()); print(df.describe(include='all'))
print(df.isna().sum())

num = df.select_dtypes('number').columns
cat = df.select_dtypes('object').columns
df[num]=df[num].fillna(df[num].median())
df[cat]=df[cat].fillna('unkown')

q1,q3 = df['Fare'].quantile([.25,.75]); iqr=q3-q1
lo,hi=q1-1.5*iqr,q3+1.5*iqr
df['Fare']=df['Fare'].clip(lo,hi)

X=pd.get_dummies(df.drop(columns=['Survived']),drop_first=True)
y=df['Survived']

X_tr,X_tmp,y_tr,y_tmp=train_test_split(X,y,test_size=.3,stratify=y,random_state=42)
X_va,X_te,y_va,y_te=train_test_split(X_tmp,y_tmp,test_size=.5,stratify=y_tmp,random_state=42)

print('--------------------------')
print(df.info()); print(df.describe(include='all'))
print(df.isna().sum())

