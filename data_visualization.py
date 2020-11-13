import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_cleaning import helper, get_cabin
import warnings


warnings.filterwarnings('ignore')
sns.set(style="white", color_codes=True)


def plotting(data):
    data['Survived'] = data['Survived'].map({
        0: 'Died',
        1: 'Survived'
    })

    data['Embarked'] = data['Embarked'].map({
        'C': 'Cherbourg',
        'Q': 'Queenstown',
        'S': 'Southampton',
    })

    ax = sns.countplot(x = 'Pclass', hue = 'Survived', palette = 'hls', data = data)
    ax.set(title = 'Passenger status (Survived/Died) against Passenger Class', xlabel = 'Passenger Class', ylabel = 'Total')
    plt.show()

    ax = sns.countplot(x = 'Sex', hue = 'Survived', palette = 'hls', data = data)
    ax.set(title = 'Total Survivors According to Sex', xlabel = 'Sex', ylabel='Total')
    plt.show()

    interval = (0, 15, 30, 45, 60, 75, 90)
    categories = ['0~15','15~30','30~45', '45~60', '60~75','75~90']
    data['Age_cats'] = pd.cut(data.Age, interval, labels = categories)
    ax = sns.countplot(x = 'Age_cats',  data = data, hue = 'Survived', palette = 'hls')
    ax.set(xlabel='Age Categorical', ylabel='Total', title="Age Categorical Survival Distribution")
    plt.show()

    ax = sns.countplot(x = 'SibSp', hue = 'Survived', palette = 'hls', data = data)
    ax.set(title = 'Survival distribution according to Number of siblings/spouses on board')
    plt.show()

    ax = sns.countplot(x = 'Parch', hue = 'Survived', palette = 'hls', data = data)
    ax.set(title = 'Survival distribution according to number of parents/children on board')
    plt.show()

    interval = (0, 15, 30, 45, 100, 300, 550)
    categories = ['0~15','15~30','30~45', '45~100', '100~300','300~550']
    data['Fare_cats'] = pd.cut(data.Fare, interval, labels = categories)
    ax = sns.countplot(x = 'Fare_cats', data = data, hue = 'Survived', palette = 'hls')
    ax.set(title = 'Survival distribution according to Passenger fare')
    plt.show()

    df = data['Cabin']
    data['Cabin_cats'] = df.astype(str).str[0]
    ax = sns.countplot(x = 'Cabin_cats', data = data, hue = 'Survived', palette = 'hls')
    ax.set(title = 'Survival distribution according to Cabin no')
    plt.show()

    ax = sns.countplot(x = 'Embarked', hue = 'Survived', palette = 'hls', data = data)
    ax.set(title = 'Survival distribution according to Embarking place')
    plt.show()

    data_ = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Age_cats', 'Fare_cats'], 1)
    data_.Sex.replace(('male','female'), (0,1), inplace = True)
    data_.Embarked.replace(('Southampton','Cherbourg','Queenstown'), (0,1,2), inplace = True)
    data_.Survived.replace(('Died','Survived'), (0,1), inplace = True)
    data_.Cabin_cats.replace(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'n'), (1,2,3,4,5,6,7,8,-1), inplace = True)
    data_.Cabin_cats.fillna(0, inplace=True)
    # print(data_.head())

    plt.figure(figsize=(14,12))
    sns.heatmap(data_.astype(float).corr(),linewidths=0.1, square=True,  linecolor='white', annot=True)
    plt.show()


def data_features():
    data = pd.read_csv("data/titanic/train.csv", header=0)
    TOTAL = len(data)

    cabin_list = get_cabin(data)
    data__ = pd.DataFrame([helper(data, i, cabin_list, 'train') for i in range(TOTAL)],
                         columns=["PassengerID", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "Survived"])
    data__.drop(['PassengerID', 'Survived'], 1, inplace=True)
    train_feature = pd.DataFrame(data={'NumSam': [], 'Mean': [], 'Variance': []})
    train_feature.NumSam = data__[data__ != -1].count()
    train_feature.Mean = data__.mean()
    train_feature.Variance = data__.var()
    print(train_feature)
    train_feature.to_csv("data/data_visualization/train_data_feature.csv")

    data = pd.read_csv("data/titanic/test.csv", header=0)
    TOTAL = len(data)


    cabin_list = get_cabin(data)
    data__ = pd.DataFrame([helper(data, i, cabin_list, 'test') for i in range(TOTAL)],
                          columns=["PassengerID", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"])
    data__.drop(['PassengerID'], 1, inplace=True)
    train_feature = pd.DataFrame(data={'NumSam': [], 'Mean': [], 'Variance': []})
    train_feature.NumSam = data__[data__ != -1].count()
    train_feature.Mean = data__.mean()
    train_feature.Variance = data__.var()
    print(train_feature)
    train_feature.to_csv("data/data_visualization/test_data_feature.csv")

    cabin_list = get_cabin(data)
    data__ = pd.DataFrame([helper(data, i, cabin_list, 'test') for i in range(TOTAL)],
                          columns=["PassengerID", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin",
                                   "Embarked"])
    data__.drop(['PassengerID'], 1, inplace=True)
    train_feature = pd.DataFrame(data={'NumSam': [], 'Mean': [], 'Variance': []})
    train_feature.NumSam = data__[data__ != -1].count()
    train_feature.Mean = data__.mean()
    train_feature.Variance = data__.var()
    print(train_feature)
    train_feature.to_csv("data/data_visualization/test_data_feature.csv")



def prediction_eval():
    df_prediction = pd.read_csv('data/titanic/prediction.csv')
    df_prediction = df_prediction[df_prediction.Survived == 1]
    print(len(df_prediction), 'survivors classified')
    print(len(df_prediction[df_prediction.Sex == 'female']), 'survived females passengers')
    print(len(df_prediction[df_prediction.Age < 18]), 'survived passengers under 18')
    print(len(df_prediction[df_prediction.SibSp == 0][df_prediction.Parch == 0]),
          'survived passengers do not have any family member on board')

    temp = min(len(df_prediction[df_prediction.Pclass == 1]),
               len(df_prediction[df_prediction.Pclass == 2]),
               len(df_prediction[df_prediction.Pclass == 3]))

    str = 'have the least chance of surviving the tragedy'
    if len(df_prediction[df_prediction.Pclass == 1]) == temp: print('Ticket class 1', str)
    elif len(df_prediction[df_prediction.Pclass == 2]) == temp: print('Ticket class 2', str)
    else: print('Ticket class 3', str)

    temp = min(len(df_prediction[df_prediction.Embarked == 'C']),
               len(df_prediction[df_prediction.Embarked == 'Q']),
               len(df_prediction[df_prediction.Embarked == 'S']))

    if len(df_prediction[df_prediction.Embarked == 'C']) == temp: print('Cherbourg', str)
    elif len(df_prediction[df_prediction.Embarked == 'Q']) == temp: print('Queenstown', str)
    else: print('Southampton', str)



def result_comparison():
    df_ = pd.read_csv('data/titanic/result_comparison.csv', header=0)
    list_ = ['survivor', 'female', 'eighteen', 'family', 'class']
    for i in range(len(list_)):
        print(df_[list_[i]].mean())
        print(df_[list_[i]].var())


# data_features()
# plotting(pd.read_csv("data/titanic/train.csv", header=0))
# prediction_eval()
# result_comparison()