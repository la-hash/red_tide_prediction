import dgl
import numpy
import pandas as pd
import torch as th
import random
import numpy as np
from torch.autograd import Variable
from readfiles import preprocess
import warnings

# action参数可以设置为ignore，一位一次也不行，once表示为只显示一次
warnings.filterwarnings(action='ignore')


def future_data_every_list():
    data_list = ['0622', '0624', '0628', '0629', '0704', '0707', '0708', '0711', '0712', '0713', '0714', '0715',
                 '0716',
                 '0718', '0721', '0722', '0726', '0727', '0728', '0801', '0802', '0805']

    normaled_file_features_frame = pd.read_csv(
        'E:\\Unzip file\\temporal-gcn-lstm-master\\normaled_file_features_frame.csv')

    future_data_every_list = [[[] for _ in range(4)] for _ in range(len(data_list))]

    count = 0
    for i, data in enumerate(data_list):
        result = []
        nums = 0
        for m in range(count, len(normaled_file_features_frame)):
            nums += 1
            aa = (normaled_file_features_frame['data'][m])
            data = int(data)
            if aa == data:
                temp_future = normaled_file_features_frame.iloc[m, :].tolist()
                # print('temp_future',temp_future)
                result.append(temp_future[1:])
            else:
                break
        future_data = pd.DataFrame(result,
                                   columns=['data', 'location', 'Class', 'number', 'MCI', 'ThreeBand', 'Index1',
                                            'Index2', 'InfraredPeak'])
        # print(future_data)

        count += nums - 1

        columns = ['data', 'location', 'Class', 'number', 'MCI', 'ThreeBand', 'Index1',
                   'Index2', 'InfraredPeak']

        temp_future_WC = future_data.iloc[(future_data['location'] == 'WC').tolist()].to_numpy()
        temp_future_EC = future_data.iloc[(future_data['location'] == 'EC').tolist()].to_numpy()
        temp_future_EF = future_data.iloc[(future_data['location'] == 'EF').tolist()].to_numpy()
        temp_future_WF = future_data.iloc[(future_data['location'] == 'WF').tolist()].to_numpy()

        future_data_WC = pd.DataFrame(temp_future_WC, columns=columns)
        future_data_EC = pd.DataFrame(temp_future_EC, columns=columns)
        future_data_EF = pd.DataFrame(temp_future_EF, columns=columns)
        future_data_WF = pd.DataFrame(temp_future_WF, columns=columns)
        # print(future_data_WC)
        # print(future_data_EC)
        # print(future_data_EF)
        # print(future_data_WF)

        future_data_every_list[i][0] = future_data_WC  # WC
        future_data_every_list[i][1] = future_data_EC  # EC
        future_data_every_list[i][2] = future_data_EF  # EF
        future_data_every_list[i][3] = future_data_WF  # WF

    # print(future_data_every_list)
    return future_data_every_list

MCI_median_list = [0.2310171348481267, 0.2818856744335314, 0.3458505832991972,
 0.3234866489745737, 0.3392941469799549 ,0.3522695532643146,
 0.3474758342609771, 0.3205842060165604 ,0.2949294289298764,
 0.097376534066157, 0.1029986802807835]

ThreeBand_median_list = [0.1093607618981376, 0.1510103605082429, 0.1921128630466536,
 0.1722587582870559, 0.182018984631191, 0.1909639694904152,
 0.1811256683950337, 0.1897456860179867, 0.1525263812790066,
 0.0306273267054187, 0.0332527311736541]

Index1_median_list = [0.3733794088371096, 0.4176764375549895, 0.4512870173634423,
 0.4286710342978809, 0.4426467028125165, 0.4494526626540571,
 0.4501753569403932, 0.5074546299593046, 0.4161867970916884,
 0.1976972897991528, 0.2115865815608567]

Index2_median_list = [0.3569149928577222, 0.3556882314172429, 0.3562095555706372,
 0.3568124451351827, 0.3576932742864487, 0.3563762813903805,
 0.3567625559913503, 0.3566921624528177, 0.356062412869959,
 0.3553317671841683, 0.3555744905478077]

InfraredPeak_median_list = [0.4019975208875542, 0.4016019402635776, 0.4014581856029948,
 0.4014564277289882, 0.4016732479794305, 0.4018496129227644,
 0.4020613543867795, 0.4019740610173619, 0.4019030055618756,
 0.4016089269886099, 0.4024766165286917]

def combination(future_data_every_list, train_volum):

    future_data_selected = [[[] for _ in range(2)] for _ in range(train_volum)]

    for i in range(train_volum):

        index_everyday = [[[ ]for _ in range(4) ]for _ in range(10) ]
        start_day = random.randint(0, len(future_data_every_list) - 1-11)
        for j,day in enumerate(range(start_day ,start_day + 10)):
            df_WC = future_data_every_list[day][0]
            df_EC = future_data_every_list[day][1]
            df_EF = future_data_every_list[day][2]
            df_WF = future_data_every_list[day][3]

            p_WC, _ = df_WC.shape
            p_EC, _ = df_EC.shape
            p_EF, _ = df_EF.shape
            p_WF, _ = df_WF.shape

            if p_WC==0:
                # print('day')
                # print(day)
                MCI_random = random.choice(MCI_median_list)
                ThreeBand_random = random.choice(ThreeBand_median_list)
                Index1_random = random.choice(Index1_median_list)
                Index2_random = random.choice(Index2_median_list)
                index_everyday[j][0] = [MCI_random,ThreeBand_random,Index1_random,Index2_random]
            else:
                WC_number =random.randint(0, p_WC - 1)
                # print('WC_number', WC_number)
                index_everyday[j][0] = df_WC.iloc[WC_number, 4:8].tolist()
            # print('*' * 50)
            # print(index_everyday[j][0])

            if p_EC==0:
                MCI_random = random.choice(MCI_median_list)
                ThreeBand_random = random.choice(ThreeBand_median_list)
                Index1_random = random.choice(Index1_median_list)
                Index2_random = random.choice(Index2_median_list)
                index_everyday[j][1] = [MCI_random, ThreeBand_random, Index1_random, Index2_random]
            else:
                EC_number =random.randint(0, p_EC - 1)
                # print('EC_number',EC_number)
                index_everyday[j][1] = df_EC.iloc[EC_number, 4:8].tolist()

            # print('*'*50)
            # print(index_everyday[j][1])

            if p_EF==0:
                MCI_random = random.choice(MCI_median_list)
                ThreeBand_random = random.choice(ThreeBand_median_list)
                Index1_random = random.choice(Index1_median_list)
                Index2_random = random.choice(Index2_median_list)
                index_everyday[j][2] = [MCI_random, ThreeBand_random, Index1_random, Index2_random]
            else:
                EF_number =random.randint(0, p_EF - 1)
                # print('EF_number', EF_number)
                index_everyday[j][2] = df_EF.iloc[EF_number, 4:8].tolist()
            # print('*' * 50)
            # print(index_everyday[j][2])

            if p_WF==0:
                MCI_random = random.choice(MCI_median_list)
                ThreeBand_random = random.choice(ThreeBand_median_list)
                Index1_random = random.choice(Index1_median_list)
                Index2_random = random.choice(Index2_median_list)
                index_everyday[j][3] = [MCI_random, ThreeBand_random, Index1_random, Index2_random]
            else:
                WF_number =random.randint(0, p_WF - 1)
                # print('WF_number', WF_number)
                index_everyday[j][3] = df_WF.iloc[WF_number, 4:8].tolist()
            # print('*' * 50)
            # print(index_everyday[j][3])

        future_data_selected[i][0] = index_everyday

        df_WC_label = future_data_every_list[start_day + 10][0]
        df_EC_label = future_data_every_list[start_day + 10][1]
        df_EF_label = future_data_every_list[start_day + 10][2]
        df_WF_label = future_data_every_list[start_day + 10][3]

        p_WC_label, _ = df_WC_label.shape
        p_EC_label, _ = df_EC_label.shape
        p_EF_label, _ = df_EF_label.shape
        p_WF_label, _ = df_WF_label.shape

        labels = []

        if p_WC_label == 0:
            labels.append(int(0))
        else:
            labels.append(int(df_WC_label.iloc[:,2][0]))

        if p_EC_label == 0:
            labels.append(int(0))
        else:
            labels.append(int(df_EC_label.iloc[:,2][0]))

        if p_EF_label == 0:
            labels.append(int(0))
        else:
            labels.append(int(df_EF_label.iloc[:,2][0]))

        if p_WF_label == 0:
            labels.append(int(0))
        else:
            labels.append(int(df_WF_label.iloc[:,2][0]))


        future_data_selected[i][1] = labels#改正

    return future_data_selected

def creat_graph(future_data_selected, distance):
    bg = [[[] for _ in range(2)] for _ in range(len(future_data_selected))]
    bg_frame = pd.DataFrame(bg,columns={'graph','label'})

    for i in range(len(future_data_selected)):
        future = future_data_selected[i][0]

        garph_sequence = bg_frame.iloc[i, 0]
        label_sequence = bg_frame.iloc[i, 1]
        for j in range(len(future)):
            u, v = th.tensor([0, 1, 2, 3, 1, 2]), th.tensor([1, 2, 3, 0, 3, 0])
            g_single = dgl.graph((u, v))
            bg_single = dgl.to_bidirected(g_single)
            bg_single.ndata['h'] = th.tensor(future[j],dtype=th.float32)
            bg_single.edata['w'] = th.tensor(distance,dtype=th.float32)
            garph_sequence.append(bg_single)
        if 1 in future_data_selected[i][1]:
            bg_frame.iloc[i,1] = 1
        else:
            bg_frame.iloc[i,1] = 0

    X_graph = np.array(bg_frame.iloc[:,0].values)
    Y_graph = np.array(bg_frame.iloc[:,1].values)
    return X_graph,Y_graph,bg_frame


def create_data(future_data_selected,train_volume):
    X_set = []
    Y_set = []
    for i in range(len(future_data_selected)):
        x_single = future_data_selected[i][0]
        y_single = future_data_selected[i][1]
        X_set.append(x_single)
        if 1 in y_single:
            Y_set.append(int(1))
        else:
            Y_set.append(int(0))
        # Y_set.append(y_single)
    X = np.array(X_set)
    X = X.reshape(train_volume,10,-1)

    Y = np.array(Y_set)
    # Y = Y.reshape(11,-1)
    return X,Y


# 填补空缺值（时间的空缺）时间步长不同
def weighting(a,b,c,d):
    result = -0.28*a+0.40*b-0.12*c+0.4*d-1.78
    return result

# 宏观数据，用啥啊。第一篇的加权和
def macro_each_day(future_data_selected,train_volum):
    macro_data = [[[0 for _ in range(4)] for _ in range(10)]for _ in range(train_volum)]
    for i in range(len(future_data_selected)):
        index_everyday = future_data_selected[i][0]
        for j in range(len(index_everyday)):
            WC = index_everyday[j][0]
            EC = index_everyday[j][1]
            EF = index_everyday[j][2]
            WF = index_everyday[j][3]

            macro_data[i][j][0] = weighting(WC[0],WC[1],WC[2],WC[3])
            macro_data[i][j][1] = weighting(EC[0],EC[1],EC[2],EC[3])
            macro_data[i][j][2] = weighting(EF[0],EF[1],EF[2],EF[3])
            macro_data[i][j][3] = weighting(WF[0],WF[1],WF[2],WF[3])
    macro_data_array_ = np.array(macro_data)
    macro_list = [[0 for _ in range(4)]for _ in range(len(macro_data_array_))]
    macro_data_array_transpose = macro_data_array_.transpose(0,2,1)
    for m in range(len(macro_data_array_transpose)):
        macro_list[m][0] = np.mean(macro_data_array_transpose[m][0])
        macro_list[m][1] = np.mean(macro_data_array_transpose[m][1])
        macro_list[m][2] = np.mean(macro_data_array_transpose[m][2])
        macro_list[m][3] = np.mean(macro_data_array_transpose[m][3])
    macro_data_array = np.array(macro_list)
    return macro_data_array


if __name__ == '__main__':
    future_data_every_list = future_data_every_list()
    future_data_selected = combination(future_data_every_list, 400)
    distance = [1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1]
    _,_,bg =  creat_graph(future_data_selected, distance)
    X, Y = create_data(future_data_selected,10)
    print('X', X)
    print('Y', Y)


# 静态、动态图
# 貌似我也没有静态动态的区别啊，难道把水流方向也算上吗
# def flow_direction():
