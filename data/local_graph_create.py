#图像分割
#特征提取 表格生成
import numpy as np
import gdal


def image_cut(image_arr,cut_size):
    # image_read = gdal.open(filename)
    # image_arr = image_read.ReadAsArray()
    # bands = image_read.RasterCount()  # 获取波段数
    # xsize,ysize = image_read.RasterXSize, image_read.RasterYSize
    ysize,xsize = image_arr.shape # 获取影像的宽高
    #影像pad填充
    pad_across = cut_size-(xsize%cut_size)
    pad_vertical = cut_size - (ysize % cut_size)
    a = np.zeros((ysize, pad_across))
    image_arr = np.concatenate((image_arr,a),axis=1)
    b = np.zeros((pad_vertical, xsize + pad_across))
    image_arr = np.concatenate((image_arr, b), axis=0)
    #划分子图
    downsize = [[[] for _ in range(xsize//cut_size) ] for _ in range(ysize//cut_size)]#定义空列表
    for i in range(xsize//cut_size):
        for j in range(ysize//cut_size):
            downsize[i][j] = image_arr[cut_size*j:cut_size*(j+1)-1,cut_size*i:cut_size*(i+1)-1]
    return downsize
    #返回列表，列表为m*n，
    # 其中每个元素还是列表，里面存储着一个子图，格式为array


def hy1c_BandFeatures(filename,):#可以copy做一个sentinel2或3的
    image_read = gdal.open(filename)

    # bands_count = image_read.RasterCount()  # 获取波段数
    # bands = []#将波段存储到列表
    # for i in range(bands_count):
    #     bands[i] = image_read.GetRasterBand(i)

    #获取波段
    #蓝波段
    blue = image_read.GetRasterBand(1).ReadAsArray() * 0.0001
    # 绿波段
    gre = image_read.GetRasterBand(2).ReadAsArray() * 0.0001
    # 红波段
    red = image_read.GetRasterBand(3).ReadAsArray() * 0.0001
    # 近红外波段
    nir = image_read.GetRasterBand(4).ReadAsArray() * 0.0001

    #计算特征
    #赤潮值
    redtide_index =
    #叶绿素
    chlorophyll_arr =
    #RI指数
    RI =
    #辅助指标


    # NAN——>0
    nan_index = np.isnan(ndvi)
    ndvi[nan_index] = 0
    ndvi = ndvi.astype(np.float32)#对每个特征重复


    #特征连接 查：python array 拼接 最终输出为normaled_file_features的形式


