import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from data_cleaning import load_data
from tensorboardX import SummaryWriter
from data_visualization import prediction_eval



def train(config, model, train_vectors, train_labels, dev_vectors, dev_labels):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum = config.momentum)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        for start in range(0, len(train_vectors), config.batch_size):
            end = start + config.batch_size
            batch_vectors = train_vectors[start:end]
            batch_labels = train_labels[start:end]

            outputs = model(batch_vectors)
            model.zero_grad()  # grad set to 0
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()  # compute gradient
            optimizer.step()  # update parameter

            if total_batch % 5 == 0:  # 每多少轮输出在训练集和验证集上的效果
                true = batch_labels.data
                predic = torch.max(outputs.data, 1)[1]
                # print(predic)
                train_acc = accuracy_score(true, predic)
                dev_acc, dev_loss, _ = evaluate(config, model, dev_vectors, dev_labels)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
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
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    accuracy = accuracy_score(train_labels, torch.max(model(train_vectors).data,1)[1])
    print("Training Accuracy:", accuracy)
    accuracy, _, confusion = evaluate(config, model, dev_vectors, dev_labels)
    print("Test Accuracy:", accuracy)
    print("Confusion Matrix:", )
    print(confusion)

    writer.close()

    gold_vectors = load_data('test')
    gold_prediction = torch.max(model(gold_vectors).data, 1)[1].numpy()
    # print(gold_prediction)
    df_prediction = pd.read_csv('data/titanic/test.csv', header=0)
    df_prediction['Survived'] = gold_prediction
    df_prediction.to_csv("data/titanic/prediction.csv")

    prediction_eval()



def evaluate(config, model, data_vectors, data_labels):
    model.eval()  # lock batch normalisation & dropout
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for start in range(0, len(data_vectors), config.batch_size):
            end = start + config.batch_size
            batch_vectors = data_vectors[start:end]
            batch_labels = data_labels[start:end]

            outputs = model(batch_vectors)
            loss = F.cross_entropy(outputs, batch_labels) 
            loss_total += loss
            batch_labels = batch_labels.data.numpy()
            predic = torch.max(outputs.data, 1)[1].numpy()
            labels_all = np.append(labels_all, batch_labels)
            predict_all = np.append(predict_all, predic)
    acc = accuracy_score(labels_all, predict_all)
    confusion = confusion_matrix(labels_all, predict_all)
    return acc, loss_total / len(data_vectors), confusion



