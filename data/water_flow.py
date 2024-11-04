import numpy as np




#珠江口水域裁剪

#0值填充
#水流信息嵌入部（查询）查询标准字段，询问海流标准格式，毕竟是为海流设定的

def point_fill0(feature):#feature为array
    # NAN——>0
    nan_index = np.isnan(feature)
    feature[nan_index] = 0




def waterflow_information_array():
