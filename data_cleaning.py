import pandas as pd
import torch
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/titanic/train.csv', header=0)
TOTAL = len(data)


def get_cabin():
    cabin_list = []
    for i in range(TOTAL):
        if not pd.isnull(data['Cabin'][i]):
            list = data['Cabin'][i].split()
            for j in range(len(list)):
                if list[j] not in cabin_list:
                    cabin_list.append(list[j])

    return sorted(cabin_list)


cabin_list = get_cabin()
# dropping unecessary parameters
def helper(i):
    ans = [0] * 10
    ans[0] = data['PassengerId'][i]
    ans[1] = data['Pclass'][i]
    ans[4] = data['SibSp'][i]
    ans[5] = data['Parch'][i]
    ans[6] = data['Fare'][i]

    if data['Sex'][i] == 'male': ans[2] = 1
    else: ans[2] = 2

    if not pd.isnull(data['Age'][i]): ans[3] = data['Age'][i]

    if not pd.isnull(data['Cabin'][i]):
        list = data['Cabin'][i].split()
        ans[7] = torch.mean(torch.FloatTensor([cabin_list.index(list[j]) for j in range(len(list))]))

    if data['Embarked'][i] == 'C': ans[8] = 1
    elif data['Embarked'][i] == 'Q': ans[8] = 2
    elif data['Embarked'][i] == 'S': ans[8] = 3

    ans[9] = data['Survived'][i]

    return ans


def load_data():
    trX = torch.Tensor([helper(i)[0:8] for i in range(TOTAL)])
    trY = torch.Tensor([helper(i)[9] for i in range(TOTAL)])
    # print(trX, trY)

    # Random Splitting
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(trX, trY, test_size=0.15)
    return(train_vectors, test_vectors, train_labels, test_labels)

# load_data()