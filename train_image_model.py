import numpy as np
import Image as I
from ikrlib import png2fea
import torch
import torch.nn.functional as F
import copy


def eval_fin(m, t_t, n_t, t_d, n_t_d):
    correct = 0
    incorrect = 0
    with torch.no_grad():
        for x in t_t:
            if threshold(m(torch.FloatTensor(x))):
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        for x in n_t:
            if threshold(m(torch.FloatTensor(x))):
                incorrect = incorrect + 1
            else:
                correct = correct + 1

    correct = correct + eval_training(m, t_d, n_t_d)

    return correct


def eval_training(m, t_d, n_t_d):
    correct = 0
    incorrect = 0
    with torch.no_grad():
        for x in t_d:
            out = m(torch.FloatTensor(x))
            if threshold(out):
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        for x in n_t_d:
            out = m(torch.FloatTensor(x))
            if threshold(out):
                incorrect = incorrect + 1
            else:
                correct = correct + 1

    return correct


def threshold(x, t=0.225):
    if x < t:
        return 0
    else:
        return 1


def load_model(name):
    m = I.Model()
    m.load_state_dict(torch.load(name))
    m.eval()
    return m


def save_model(m, name):
    model_name = name
    torch.save(m.state_dict(), model_name)
    print("Saved new model to " + model_name + " file")


if __name__ == '__main__':
    train_t = np.r_[[i.mean(axis=2) for i in png2fea("target_train").values()]].reshape(-1, 80 * 80)
    train_n = np.r_[[i.mean(axis=2) for i in png2fea("non_target_train").values()]].reshape(-1, 80 * 80)
    target_dev = np.r_[[i.mean(axis=2) for i in png2fea("target_dev").values()]].reshape(-1, 80 * 80)
    non_target_dev = np.r_[[i.mean(axis=2) for i in png2fea("non_target_dev").values()]].reshape(-1, 80 * 80)

    train = list()
    for d in train_t:
        train.append((torch.Tensor(d), 1))
    for d in train_n:
        train.append((torch.Tensor(d), 0))

    trainset = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True)

    print("Traning IMAGE_MODEL")
    ####################################
    #       Training IMAGE_MODEL       #
    ####################################
    loss = 0
    loss_array = list()
    best_model = I.Model()
    model = I.Model()
    learning_rate = 5e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = 0
    for i in range(250):
        for data in trainset:
            X, y = data
            model.zero_grad()
            y_pred = model(torch.FloatTensor(X))
            loss = F.binary_cross_entropy(y_pred.squeeze(1), y.float())
            loss.backward()
            optimizer.step()
        new_score = eval_training(model, target_dev, non_target_dev)
        if new_score > best:
            best_model = copy.deepcopy(model)
            best = new_score
        if i % 5 == 4:
            print("Iteration: ", i, "\nLoss: ", loss.item(), "\nCorrect: ", best)
        loss_array.append(loss.item())

    eval_fin(best_model, train_t, train_n, target_dev, non_target_dev)
    save_model(best_model, "IMAGE_MODEL")

    print("\nTraning IMAGE_MODEL_PLUS")
    ########################################
    #       Training IMAGE_MODEL_PLUS      #
    ########################################
    learning_rate = 5e-6
    tune = list()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for x in target_dev:
        tune.append((torch.Tensor(x), 1))
    for x in non_target_dev:
        tune.append((torch.Tensor(x), 0))
    tuneset = torch.utils.data.DataLoader(tune, batch_size=5, shuffle=True)

    best = 0
    for i in range(50):
        for data in tuneset:
            X, y = data
            model.zero_grad()
            y_pred = model(torch.FloatTensor(X))
            loss = F.binary_cross_entropy(y_pred.squeeze(1), y.float())
            loss.backward()
            optimizer.step()
        new_score = eval_training(model, target_dev, non_target_dev)
        if new_score > best:
            best_model = copy.deepcopy(model)
            best = new_score
        if i % 5 == 4:
            print("Iteration: ", i, "\nLoss: ", loss.item(), "\nCorrect: ", best)

    eval_fin(best_model, train_t, train_n, target_dev, non_target_dev)
    save_model(best_model, "IMAGE_MODEL_PLUS")

    exit(0)

