from os import listdir
import pandas as pd
import os


# labelList = []  # 类标签列表
#
# datasetList = listdir("ridetide_csv")
#
# print(datasetList)
#
# datasetLength = len(datasetList)  # 文件夹中文件数量
# print(datasetLength)
# for i in range(datasetLength):
#     filename = datasetList[i]  # 获取文件名字符串
#     file = filename.split('.')[0]  # 以 . 分割提取文件名
#     classOrder = file.split('_')[0] # 以 _ 分割提取类别号
#     labelList.append(classOrder)
# print(labelList)
# #读取文件部分

Index_All_frame = pd.DataFrame(columns=['Class','MCI','ThreeBand','Index1','Index2','InfraredPeak'])

a=(709-681)/(754-681)

# f = open("Index_All.csv", "w")

file_dir = "ridetide_csv"
all_file_list = os.listdir(file_dir)
for single_file in all_file_list:
    # 逐个读取
    single_data_frame = pd.read_csv(
        os.path.join(file_dir, single_file), sep='\t', header=0)
    single_data_frame.columns = ['Wavelength', 'rt']#修改列名
    #print(single_data_frame)
#单个读取文件（已成功）


    filename = os.path.basename(single_file)
    file = filename.split('.')[0]  # 以 . 分割提取文件名
    Class = file.split('_')[0]  # 以 _ 分割提取类别号
    #print(Class)
    #类别标签读取l

    # 计算光谱指数表
    for i in range(len(single_data_frame)):
        #计算MCI
        if str(single_data_frame['Wavelength'][i]) == "681":
            rt681 = single_data_frame['rt'][i]
        else:
            pass
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "709":
            rt709 = single_data_frame['rt'][i]
        else:
            pass
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "754":
            rt754 = single_data_frame['rt'][i]
        else:pass
    MCI=rt709-rt681-a*(rt754-rt681)
    #计算三波段
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "671":
            rt671 = single_data_frame['rt'][i]
        else:
            pass
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "710":
            rt710 = single_data_frame['rt'][i]
        else:
            pass
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "754":
            rt754 = single_data_frame['rt'][i]
        else:pass
    ThreeBand=(1/rt671-1/rt710)*rt754
    # 计算指标一
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "709":
            rt709 = single_data_frame['rt'][i]
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "560":
            rt560 = single_data_frame['rt'][i]
    Index1 = (rt709 - rt560)/rt560
    # 计算指标二
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "940":
            rt940 = single_data_frame['rt'][i]
        else:
            pass
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "900":
            rt900 = single_data_frame['rt'][i]
        else:
            pass
    Index2 = (rt940 - rt900)/rt940
        # 计算红外尖峰
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "1085":
            rt1085 = single_data_frame['rt'][i]
        else:
            pass
    for i in range(len(single_data_frame)):
        if str(single_data_frame['Wavelength'][i]) == "980":
            rt980 = single_data_frame['rt'][i]
        else:
            pass
    InfraredPeak=rt1085/rt980
        #添加进总表

    # df = df.append({'A': i}, ignore_index=True)
    Index_All_frame = Index_All_frame.append({'Class':Class,'MCI':MCI,'ThreeBand':ThreeBand,'Index1':Index1,'Index2':Index2,'InfraredPeak':InfraredPeak}, ignore_index=True)
    # f.writelines({'Class':Class,'MCI':MCI,'ThreeBand':ThreeBand,'Index1':Index1,'Index2':Index2,'InfraredPeak':InfraredPeak}+'\n')

# f.close()
print(Index_All_frame)
# print(format(Index_All))
# with open("Index_All.csv", "w") as opfile:
#     opfile.write("\n".join(Index_All))
Index_All_frame.to_csv("Index_All.csv")

#     if single_file == all_file_list[0]:
#         all_data_frame = single_data_frame
#     else:  # 进行concat操作
#         all_data_frame = pd.concat([all_data_frame,
#                                     single_data_frame], ignore_index=True)
# print(all_data_frame)
# 把文件连接成一整个

