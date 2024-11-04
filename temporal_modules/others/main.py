import pandas as pd
import numpy as np
from tqdm import tqdm
import torch as th
import dgl
import warnings
# action参数可以设置为ignore，一位一次也不喜爱你是，once表示为只显示一次
warnings.filterwarnings(action='ignore')


# u, v = torch.tensor([0, 0, 0, 0, 0]), torch.tensor([1, 2, 3, 4, 5])
# g3 = dgl.graph((u, v))

# u, v = th.tensor([0, 1, 2, 3, 1, 2]), th.tensor([1, 2, 3, 0, 3, 0])
# # u, v = [0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 3, 0]
# g_single = dgl.graph((u, v))
# print(g_single)
def a():

    dic = {
        'a':[1,2,3,3,4],
        'b':[1,2,3,3,4],
        'c':[1,2,3,3,4],
        'd':[1,2,3,3,4]
           }

    df = pd.DataFrame(dic)
    return  df


if __name__ == '__main__':
    data_list = ['0622', '0624', '0628', '0629', '0704', '0707', '0708', '0711', '0712', '0713', '0714', '0715', '0716',
                 '0718', '0721', '0722', '0726', '0727', '0728', '0801', '0802', '0805']
    # data_list = ['0721']
    normaled_file_features_frame = pd.read_csv('E:\\Unzip file\\temporal-gcn-lstm-master\\normaled_file_features_frame.csv')

    future_data_every = []
    count = 0
    for i, data in enumerate(data_list):
        result = []
        nums = 0
        for m in range(count,len(normaled_file_features_frame)):
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
        future_data_every.append(future_data)

        count += nums - 1
        # break
        # future_data_every.append(future_data)
    # print(future_data_every[-1])
    # print(len(future_data_every))
    # print(len(data_list))
    # print(normaled_file_features_frame)
    result = 0
    aaa = []
    for df in future_data_every:
        result += len(df)
        aaa.append(len(df))
        print(len(df))
        print(df)
        print('*'*100)
    print(aaa)
    # [50, 10, 30, 50, 40, 20, 20, 40, 80, 130, 40, 20, 30, 60, 30, 30, 40, 29, 30, 30, 20, 20]
    # print('aaaaaaaaaaaaa')
    # print(result)
    # print(len(normaled_file_features_frame))
    # print(result == len(normaled_file_features_frame))

