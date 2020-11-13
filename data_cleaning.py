import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def get_cabin(dataset):
    cabin_list = []
    for i in range(len(dataset)):
        if not pd.isnull(dataset['Cabin'][i]):
            list = dataset['Cabin'][i].split()
            for j in range(len(list)):
                if list[j] not in cabin_list:
                    cabin_list.append(list[j])

    return sorted(cabin_list)



# dropping unecessary parameters
def helper(dataset, i, cabin_list, mode):
    ans = [-1] * 10 if mode== 'train' else [-1] * 9

    ans[0] = dataset['PassengerId'][i]
    ans[1] = dataset['Pclass'][i]
    ans[2] = 1 if dataset['Sex'][i] == 'male' else 2

    if not pd.isnull(dataset['Age'][i]): ans[3] = dataset['Age'][i]

    ans[4] = dataset['SibSp'][i]
    ans[5] = dataset['Parch'][i]

    if not pd.isnull(dataset['Fare'][i]): ans[6] = dataset['Fare'][i]

    if not pd.isnull(dataset['Cabin'][i]):
        list = dataset['Cabin'][i].split()
        ans[7] = torch.mean(torch.FloatTensor([cabin_list.index(list[j]) for j in range(len(list))]))

    if dataset['Embarked'][i] == 'C': ans[8] = 1
    elif dataset['Embarked'][i] == 'Q': ans[8] = 2
    elif dataset['Embarked'][i] == 'S': ans[8] = 3

    if mode=='train': ans[9] = dataset['Survived'][i]

    return ans


def load_data(dataset):
    if dataset == 'train':
        data = pd.read_csv("data/titanic/train.csv", header=0)
        TOTAL = len(data)
        cabin_list = get_cabin(data)
        trX = torch.Tensor([helper(data, i, cabin_list, 'train')[0:9] for i in range(TOTAL)])
        trY = torch.LongTensor([helper(data, i, cabin_list, 'train')[9] for i in range(TOTAL)])
        # print(trX, trY)
        # print(trX.shape)

        # Random Splitting
        train_vectors, test_vectors, train_labels, test_labels = train_test_split(trX, trY, test_size=0.15)
        return (train_vectors, test_vectors, train_labels, test_labels)

    elif dataset == 'test':
        data = pd.read_csv("data/titanic/test.csv", header=0)
        TOTAL = len(data)
        cabin_list = get_cabin(data)
        data_ = data
        test = torch.Tensor([helper(data_, i, cabin_list, 'test') for i in range(TOTAL)])
        # print(test)
        # print(test.shape)
        return test

# load_data('train')
# load_data('test')