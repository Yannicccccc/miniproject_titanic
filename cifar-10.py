import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import argparse
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



parser = argparse.ArgumentParser(description='EE4483 mini project')
parser.add_argument('--model', type=str, required=True, help='choose a model: ')
args = parser.parse_args()



classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_epochs = 15
log_path = 'data/log/cifar-10/CNN'
save_path = 'data/saved_dict/cifar-10/CNN.ckpt'
require_improvement = 10000


def load_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # data = [d[0].data.numpy() for d in trainloader]
    # print(np.mean(data), np.std(data))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.activate = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 30, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = 0.25
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.pool(self.activate(self.conv1(x)))
        out = self.pool(self.activate(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5)
        out = self.activate(self.fc1(out))
        out = self.activate(self.fc2(out))
        out = self.fc3(out)
        return out



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.in_channels = 3
        self.VGG13 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = self._make_layers(self.VGG13)
        self.fc = nn.Linear(512, 10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.conv = nn.Conv2d(kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [self.pool1]
            else:
                layers += [self.conv(in_channels, x),
                           self.norm(x),
                           self.relu(inplace=True)]
                in_channels = x
        layers += [self.pool2]
        return nn.Sequential(*layers)




def training(model):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainloader, testloader = load_data()

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        for i, data in enumerate(trainloader, 0):
            batch_vectors, batch_labels = data

            outputs = model(batch_vectors)
            model.zero_grad()
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            if total_batch % 2000 == 0:    # print every 2000 mini-batches
                true = batch_labels.data
                predic = torch.max(outputs.data, 1)[1]
                train_acc = accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, testloader)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'

                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()  # prevent over fitting

            total_batch += 1
            if total_batch - last_improve > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break

    print('Finished Training')
    writer.close()


def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            c = (predic == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    acc = accuracy_score(labels_all, predict_all)


    if test:
        report = classification_report(labels_all, predict_all, target_names=classes, digits=4)
        confusion = confusion_matrix(labels_all, predict_all)
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

        return acc, loss_total / len(data_iter), report, confusion

    return acc, loss_total / len(data_iter)


def test(model):
    _, test_iter = load_data()
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)



if __name__ == '__main__':
    if args.model == 'VGG': model = VGG()
    else: model = CNN()
    training(model)
    test(model)