import argparse

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm

from lstm_model import LSTMs
from gcnmodel import GCN

from data_to_graph import create_data,creat_graph,macro_each_day,combination,future_data_every_list




"""Multi-channel temporal end-to-end framework"""

"""Input datasets
    df: dataframe with userid as index, contains labels, and activity sequence if needed
    macro: dataframe containing macroscopic data if needed
    graphs: dictionary with format {user_id: list of networkx graphs}}

    ** Modify acitivity and macro flags to run different versions of model to include features
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gcn_in', type=int, default=4)  # 12 in paper
    parser.add_argument('--gcn_hid', type=int, default=20)
    parser.add_argument('--gcn_out', type=int, default=2)
    parser.add_argument('--lstm_hid', type=int, default=32)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_drop', type=int, default=0)
    parser.add_argument('--a_in', type=int, default=12)  # 10 in paper
    # parser.add_argument('--macro_dim', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)  # can increase
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--period', type=int, default=10)  # 14 days in paper

    parser.add_argument('--activity', type=bool, default=True)
    parser.add_argument('--macro', type=bool, default=False)

    # parser.add_argument('--df_path', type=str, default='mock_data/example_df.pkl')
    # parser.add_argument('--macro_path', type=str, default='mock_data/example_macro_df.pkl')
    # parser.add_argument('--graphs_path', type=str, default='mock_data/example_temporal_user_action_graphs.pkl')

    args = parser.parse_args()

    # data_list = ['0622', '0624', '0628', '0629', '0704', '0707', '0708', '0711',
    #              '0712', '0713', '0714', '0715', '0716',
    #              '0718', '0721', '0722', '0726', '0727', '0728', '0801', '0802', '0805']
    # distance=[]#距离


    # 读入 future_data_selected，lstm数据
    if args.macro:
        macro = macro_each_day()#修改这句函数


    future_data_every_list = future_data_every_list()
    train_volume = 11
    future_data_selected = combination(future_data_every_list, train_volume)
    distance = [1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1]

    print(future_data_selected)
    X_graph, Y_graph,bg_frame = creat_graph(future_data_selected, distance)



    # create graph input
    dgls, inputs, xav, eye, dim = [], [], [], [], 20
    for gid in range(len(bg_frame)):
        g_list = bg_frame.iloc[gid,0]
        print(g_list)
        temp_g, temp_adj, temp_xav = [], [], []
        for G in g_list:

            g = dgl.to_networkx(G)
            temp_g.append(G)
            temp_adj.append(np.array(nx.to_numpy_matrix(g)))
            temp_xav.append(nn.init.xavier_uniform_(torch.zeros([12, dim])))
        dgls.append(temp_g)
        inputs.append(temp_adj)
        xav.append(temp_xav)



    # train, test split
    n = len(dgls)
    split = int(n * .8)
    index = np.arange(n)
    np.random.seed(32)
    np.random.shuffle(index)
    train_index, test_index = index[:split], index[split:]

    # prep labels - +1 here bc original is [-1, 0, 1]    图数据
    train_labels, test_labels = Variable(torch.LongTensor((Y_graph.astype(int))[train_index])), Variable(
        torch.LongTensor((Y_graph.astype(int))[test_index]))

    # prep temporal graph data
    k = args.period
    trainGs, testGs = [dgls[i] for i in train_index], [dgls[i] for i in test_index]
    trainGs, testGs = [dgl.batch([u[i] for u in trainGs]) for i in range(k)], \
                      [dgl.batch([u[i] for u in testGs]) for i in range(k)]
    train_inputs, test_inputs = [inputs[i] for i in train_index], [inputs[i] for i in test_index]
    train_inputs, test_inputs = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_inputs])) for i in range(k)], \
                                [torch.FloatTensor(np.concatenate([inp[i] for inp in test_inputs])) for i in range(k)]
    train_xav, test_xav = [xav[i] for i in train_index], [xav[i] for i in test_index]
    train_xav, test_xav = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_xav])) for i in range(k)], [
        torch.FloatTensor(np.concatenate([inp[i] for inp in test_xav])) for i in range(k)]

    # prep activity sequence data
    if args.activity:
        X, Y = create_data(future_data_selected,train_volume)
        x_train = Variable(torch.FloatTensor(X[train_index, :, :]), requires_grad=False)
        x_test = Variable(torch.FloatTensor(X[test_index, :, :]), requires_grad=False)
        y_train = Variable(torch.LongTensor(Y[train_index]), requires_grad=False)
        y_test = Variable(torch.LongTensor(Y[test_index]), requires_grad=False)
    # prep macro data
    #macro函数产生的数据
    if args.macro:
        macro_train = Variable(torch.FloatTensor(macro[train_index]), requires_grad=False)
        macro_test = Variable(torch.FloatTensor(macro[test_index]), requires_grad=False)

    # define models
    model = LSTMs(args.gcn_out, args.lstm_hid, args.num_classes, args.lstm_layers, args.lstm_drop)
    net = GCN(args.gcn_in, args.gcn_hid, args.gcn_out)
    model1 = LSTMs(args.a_in, args.lstm_hid, args.num_classes, args.lstm_layers, args.lstm_drop)
    linear_in_dim = args.num_classes + (args.num_classes if args.activity else 0) + (
        args.macro_dim if args.macro else 0)
    linear = nn.Linear(linear_in_dim, args.num_classes)

    parameters = list(net.parameters()) + list(model.parameters()) + list(model1.parameters()) + list(
        linear.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    dropout = nn.Dropout(args.drop)

    for epoch in tqdm(range(args.epoch)):

        # train
        model.train()
        net.train()

        # Run through GCN

        # for i in range(k):
        #     print('trainGs[i]', trainGs[i].shape)
        #     print('train_inputs[i]',train_inputs[i].shape)


        sequence = torch.stack([net(trainGs[i], train_inputs[i]) for i in range(k)], 1)
        # Temporal graph embeddings through lstm
        last, out = model(sequence)

        cat = out
        # Activity sequence through lstm
        if args.activity:
            last1, out1 = model1(x_train)
            cat = torch.cat((cat, out1), 1)
        if args.macro:
            cat = torch.cat((cat, macro_train), 1)
        cat = dropout(cat)

        mapped = linear(cat)

        logp = F.log_softmax(mapped, 1)
        loss = F.nll_loss(logp, train_labels)

        f1 = f1_score(train_labels, torch.argmax(logp, 1).data.numpy(), average='macro')

        # eval
        model.eval()
        net.eval
        test_sequence = torch.stack([net(testGs[i], test_inputs[i]) for i in range(k)], 1)
        last, out = model(test_sequence)
        cat = out
        if args.activity:
            last1, out1 = model1(x_test)
            cat = torch.cat((cat, out1), 1)
        if args.macro:
            cat = torch.cat((cat, macro_test), 1)
        mapped = linear(cat)

        test_logp = F.log_softmax(mapped, 1)
        test_f1 = f1_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Train Loss: %.4f | Train F1: %.4f | Test F1: %.4f' % (epoch, loss.item(), f1, test_f1))


        #补一个local部分预测输出