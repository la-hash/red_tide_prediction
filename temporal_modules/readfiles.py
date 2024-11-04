import os
import pandas as pd
import numpy as np
import csv
import codecs

import warnings

# action参数可以设置为ignore，一位一次也不喜爱你是，once表示为只显示一次
warnings.filterwarnings(action='ignore')

class preprocess():
    def __init__(self):
        pass

    def file_features(self,filer_name):
        file_features_frame = pd.DataFrame(
            columns=['data', 'location', 'Class', 'number','MCI', 'ThreeBand', 'Index1', 'Index2', 'InfraredPeak'])

        for root, dirs, files in os.walk(filer_name, topdown=False):
            for file in files:
                data = file.split('_')[0]
                location = file.split('_')[1]
                Class = file.split('_')[2]
                numberstr=file.split('_')[3]
                number=numberstr.split('(')[1].split(')')[0]


                f = open(os.path.join(root, file))
                f.read()
                spectrum_data_frame = pd.read_csv(os.path.join(root, file), sep='\t', header=0)  # 注：代码文件和.csv放在同一目录下
                Index = pd.DataFrame(spectrum_data_frame)
                spectrum_data_frame.columns = ['Wavelength', 'rt']

                # 计算MCI
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "681":
                        rt681 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "709":
                        rt709 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "754":
                        rt754 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                MCI = rt709 - rt681 - (709 - 681) / (754 - 681) * (rt754 - rt681)
                # 计算三波段
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "671":
                        rt671 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "710":
                        rt710 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "754":
                        rt754 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                ThreeBand = (1 / rt671 - 1 / rt710) * rt754
                # 计算指标一
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "709":
                        rt709 = spectrum_data_frame['rt'][i]
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "560":
                        rt560 = spectrum_data_frame['rt'][i]
                Index1 = (rt709 - rt560) / rt560
                # 计算指标二
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "940":
                        rt940 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "900":
                        rt900 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                Index2 = (rt940 - rt900) / rt940
                # 计算红外尖峰
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "1085":
                        rt1085 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                for i in range(len(spectrum_data_frame)):
                    if str(spectrum_data_frame['Wavelength'][i]) == "980":
                        rt980 = spectrum_data_frame['rt'][i]
                    else:
                        pass
                InfraredPeak = rt1085 / rt980

                file_features_frame = file_features_frame.append(
                    {'data': data, 'location': location, 'Class': Class,'number':number,'MCI': MCI, 'ThreeBand': ThreeBand, 'Index1': Index1, 'Index2': Index2,
                     'InfraredPeak': InfraredPeak}, ignore_index=True)
        return file_features_frame


    def normalization(self,file_features_frame,index_name):

        Index_arr = np.asarray(file_features_frame[index_name])
        index_max = max(Index_arr)
        index_min = min(Index_arr)
        normaled_feature = [0 for index in range(len(Index_arr))]  # np.array(len(Index_arr))
        for i in range(len(Index_arr)):
            normaled_feature[i] = (Index_arr[i] - index_min) / (index_max - index_min)
        normaled_feature_frame = pd.DataFrame(normaled_feature)

        return normaled_feature_frame


    def normaled_file_features_frame(self,file_features_frame):
        normaled_file_features_frame = pd.DataFrame(
           columns=['data', 'location', 'Class','number', 'MCI', 'ThreeBand', 'Index1', 'Index2', 'InfraredPeak'])
        normaled_file_features_frame['data'] = file_features_frame['data']
        normaled_file_features_frame['location'] = file_features_frame['location']
        normaled_file_features_frame['Class'] = file_features_frame['Class']
        normaled_file_features_frame['number'] = file_features_frame['number']
        normaled_file_features_frame['MCI'] = preprocess.normalization(self,file_features_frame,'MCI')
        # normaled_file_features_frame['MCI'] = self.normalization(self,file_features_frame,'MCI')
        normaled_file_features_frame['ThreeBand'] = preprocess.normalization(self,file_features_frame, 'ThreeBand')
        normaled_file_features_frame['Index1'] = preprocess.normalization(self,file_features_frame,'Index1')
        normaled_file_features_frame['Index2'] = preprocess.normalization(self,file_features_frame,'Index2')
        normaled_file_features_frame['InfraredPeak'] = preprocess.normalization(self,file_features_frame,'InfraredPeak')
        return normaled_file_features_frame

    def to_number(self, normaled_file_features_frame):
        normaled_file_features_frame['Class'][normaled_file_features_frame['Class'] == 'algea'] = 1
        normaled_file_features_frame['Class'][normaled_file_features_frame['Class'] == 'clean'] = 0
        return normaled_file_features_frame

    def future_data_every(self):
        data_list = ['0622', '0624', '0628', '0629', '0704', '0707', '0708', '0711', '0712', '0713', '0714', '0715',
                     '0716',
                     '0718', '0721', '0722', '0726', '0727', '0728', '0801', '0802', '0805']
        normaled_file_features_frame = pd.read_csv(
            'E:\\Unzip file\\temporal-gcn-lstm-master\\normaled_file_features_frame.csv')

        columns_location = ['WC', 'EC', 'EF', 'WF']
        future_data_every_block = [[[] for _ in range(4)] for _ in range(len(data_list))]
        future_data_every = pd.DataFrame(future_data_every_block, columns=columns_location)
        print(future_data_every)

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
            print(future_data)

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
            print(future_data_WC)
            print(future_data_EC)
            print(future_data_EF)
            print(future_data_WF)

            future_data_every['WC'][i] = future_data_WC
            future_data_every['EC'][i] = future_data_EC
            future_data_every['EF'][i] = future_data_EF
            future_data_every['WF'][i] = future_data_WF

        print(future_data_every)
        return future_data_every

    def future_data_every_list(self):
        data_list = ['0622', '0624', '0628', '0629', '0704', '0707', '0708', '0711', '0712', '0713', '0714', '0715',
                     '0716',
                     '0718', '0721', '0722', '0726', '0727', '0728', '0801', '0802', '0805']
        normaled_file_features_frame = pd.read_csv(
            'E:\\Unzip file\\temporal-gcn-lstm-master\\normaled_file_features_frame.csv')

        # columns_location = ['WC', 'EC', 'EF', 'WF']
        future_data_every_list = [[[] for _ in range(4)] for _ in range(len(data_list))]
        # future_data_every = pd.DataFrame(future_data_every_block, columns=columns_location)
        # print(future_data_every)

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
            print(future_data)

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
            print(future_data_WC)
            print(future_data_EC)
            print(future_data_EF)
            print(future_data_WF)

            future_data_every_list[i][0] = future_data_WC   #WC
            future_data_every_list[i][1] = future_data_EC   #EC
            future_data_every_list[i][2] = future_data_EF   #EF
            future_data_every_list[i][3] = future_data_WF   #WF

        print(future_data_every_list)
        return future_data_every_list

#     def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
#         file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
#         writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#         for data in datas:
#             writer.writerow(data)
#         print("保存文件成功，处理结束")
#
#
# def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
#     file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
#     writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#     for data in datas:
#         writer.writerow(data)
#     print("保存文件成功，处理结束")


    #还需要加一个空缺值填充模块
def combination(future_data_every_list, train_volum):
    # future_data_every=pd.read_csv()

    m, n = future_data_every.shape

    # column = ['Indexs' ,'Class']
    future_data_selected = [[[] for _ in range(2)] for _ in range(train_volum)]

    # future_data_selected = pd.DataFrame(select_table ,columns=column)

    for i in range(train_volum):

        index_everyday = [[[[ ]for _ in range(2) ]for _ in range(4) ]for _ in range(10)]
        start_day = np.random.randint(0, m - 1-11)
        for day in range(start_day ,start_day +9):
            df_WC = future_data_every[day][1]
            # df_WC = pd.DataFrame(df_WC_str,)
            print('df_WC',df_WC)
            print(type(df_WC))
            p_WC, _ = df_WC.shape
            print('行数',p_WC)

            df_EC = future_data_every[day][2]
            df_EF = future_data_every[day][3]
            df_WF = future_data_every[day][4]

            p_WC, _ = df_WC.shape
            p_EC, _ = df_EC.shape
            p_EF, _ = df_EF.shape
            p_WF, _ = df_WF.shape

            WC_number = np.random.randint(0, p_WC - 1)
            EC_number = np.random.randint(0, p_EC - 1)
            EF_number = np.random.randint(0, p_EF - 1)
            WF_number = np.random.randint(0, p_WF - 1)

            index_everyday[day][0] = df_WC.iloc[WC_number ,3:].tolist()
            index_everyday[day][1] = df_EC.iloc[EC_number, 3:].tolist()
            index_everyday[day][2] = df_EF.iloc[EF_number, 3:].tolist()
            index_everyday[day][3] = df_WF.iloc[WF_number, 3:].tolist()

        future_data_selected[i][0] = index_everyday
        future_data_selected[i][1] = [int(df_WC.iloc[start_day + 10,2]),
                                      int(df_EC.iloc[start_day + 10,2]),
                                      int(df_EF.iloc[start_day + 10,2]),
                                      int(df_WF.iloc[start_day + 10,2])]

    print(future_data_selected)

if __name__=='__main__':
    my_preprocess = preprocess()
    file_features_frame = my_preprocess.file_features("redtide_GCN")
    # print(file_features_frame)
    normaled_file_features_frame = my_preprocess.normaled_file_features_frame(file_features_frame)
    normaled_file_features_frame = my_preprocess.to_number(normaled_file_features_frame)
    # print(normaled_file_features_frame)

    normaled_file_features_frame.to_csv('normaled_file_features_frame.csv')
    future_data_every = my_preprocess.future_data_every()
    future_data_selected = combination(future_data_every, 400)
    print(future_data_selected)


    # future_data_every.to_csv('future_data_every.csv')
    #
    # future_data_every_block = my_preprocess.future_data_every_list()
    # print(future_data_every_block)
    # data_write_csv('future_data_every_list.csv',future_data_every_block)



