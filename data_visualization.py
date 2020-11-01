import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="white", color_codes=True)
data = pd.read_csv("data/titanic/train.csv", header=0)
TOTAL = len(data)

data['Survived'] = data['Survived'].map({
    0: 'Died',
    1: 'Survived'
})

data['Embarked'] = data['Embarked'].map({
    'C':'Cherbourg',
    'Q':'Queenstown',
    'S':'Southampton',
})

ax = sns.countplot(x = 'Pclass', hue = 'Survived', palette = 'hls', data = data)
ax.set(title = 'Passenger status (Survived/Died) against Passenger Class', xlabel = 'Passenger Class', ylabel = 'Total')
# plt.show()

ax = sns.countplot(x = 'Sex', hue = 'Survived', palette = 'hls', data = data)
ax.set(title = 'Total Survivors According to Sex', xlabel = 'Sex', ylabel='Total')
# plt.show()

interval = (0, 15, 30, 45, 60, 75, 90)
categories = ['0~15','15~30','30~45', '45~60', '60~75','75~90']
data['Age_cats'] = pd.cut(data.Age, interval, labels = categories)
ax = sns.countplot(x = 'Age_cats',  data = data, hue = 'Survived', palette = 'hls')
ax.set(xlabel='Age Categorical', ylabel='Total', title="Age Categorical Survival Distribution")
# plt.show()

ax = sns.countplot(x = 'SibSp', hue = 'Survived', palette = 'hls', data = data)
ax.set(title = 'Survival distribution according to Number of siblings/spouses on board')
# plt.show()

ax = sns.countplot(x = 'Parch', hue = 'Survived', palette = 'hls', data = data)
ax.set(title = 'Survival distribution according to number of parents/children on board')
# plt.show()

interval = (0, 15, 30, 45, 100, 300, 550)
categories = ['0~15','15~30','30~45', '45~100', '100~300','300~550']
data['Fare_cats'] = pd.cut(data.Fare, interval, labels = categories)
ax = sns.countplot(x = 'Fare_cats', data = data, hue = 'Survived', palette = 'hls')
ax.set(title = 'Survival distribution according to Passenger fare')
# plt.show()

df = data['Cabin'].dropna()
data['Cabin_cats'] = df.astype(str).str[0]
ax = sns.countplot(x = 'Cabin_cats', data = data, hue = 'Survived', palette = 'hls')
ax.set(title = 'Survival distribution according to Cabin no')
# plt.show()

ax = sns.countplot(x = 'Embarked', hue = 'Survived', palette = 'hls', data = data)
ax.set(title = 'Survival distribution according to Embarking place')
# plt.show()

data_ = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Age_cats', 'Fare_cats'], 1)
# print(data_)
data_.Sex.replace(('male','female'), (0,1), inplace = True)
data_.Embarked.replace(('Southampton','Cherbourg','Queenstown'), (0,1,2), inplace = True)
data_.Survived.replace(('Died','Survived'), (0,1), inplace = True)
data_.Cabin_cats.replace(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'), (1,2,3,4,5,6,7,8), inplace = True)
data_.Cabin_cats.fillna(0, inplace=True)
#print(data_.head())

plt.figure(figsize=(14,12))
sns.heatmap(data_.astype(float).corr(),linewidths=0.1, square=True,  linecolor='white', annot=True)
plt.show()
