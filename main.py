import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import  pyplot as plt

m_data = pd.read_csv('diabetes.csv')
m_cleanData = m_data.replace({
    'BloodPressure':0,
    'SkinThickness':0,
    'Insulin':0,
    'BMI':0,
    'Age':0
},m_data.mean())
inp = m_cleanData.drop(columns=['prediction'])
output = m_cleanData['prediction']
its_x_train,its_x_test,y_train,y_test = train_test_split(inp,output,test_size=0.2)
model = DecisionTreeClassifier()
model.fit(its_x_train,y_train)
predict = model.predict(its_x_test)
score = accuracy_score(y_test,predict)
print(score)
names = ['Mrs Rumela','Mrs Tripti','Mrs kaveri','Mrs Tandra','Mrs Geeta']
df = pd.read_csv('real.csv')
data = df.values.tolist()
print(data)
predict = model.predict(data)

j=0
d=0
nd=0

for i in predict:
    if i ==1:
        d =d+1
        print(names[j]," you are affected on diabetic")
    else:
        nd=nd+1
        print(names[j],"you are none diabetic")
    j=j+1

d_nd=[]
d_nd.append(d)
d_nd.append(nd)

plt.bar(names,predict)
plt.show()

y= np.array(d_nd)
mylabels = ['Diabetic','Non-Diabetic']
plt.pie(y,labels=mylabels,autopct='%1.2f%%')
plt.show()
