import os.path
import gdal
import os
import numpy as np
import gdal, osr, ogr
import glob
import gdal

import tarfile
import math

# 解压文件
def unpackage(file_name):
    # 提取解压文件夹名
    if ".tar.gz" in file_name:
        out_dir = file_name.split(".tar.gz")[0]
    else:
        out_dir = file_name.split(".")[0]
    # 进行解压
    with tarfile.open(file_name) as file:
        file.extractall(path=out_dir)
    return out_dir

#sentinel转tif格式
def S2tif(filename):
    # 打开栅格数据集
    print(filename)
    root_ds = gdal.Open(filename)
    print(type(root_ds))
    # 返回结果是一个list，list中的每个元素是一个tuple，每个tuple中包含了对数据集的路径，元数据等的描述信息
    # tuple中的第一个元素描述的是数据子集的全路径
    ds_list = root_ds.GetSubDatasets()  # 获取子数据集。该数据以数据集形式存储且以子数据集形式组织
    visual_ds = gdal.Open(ds_list[0][0])  # 打开第1个数据子集的路径。ds_list有4个子集，内部前段是路径，后段是数据信息
    visual_arr = visual_ds.ReadAsArray()  # 将数据集中的数据读取为ndarray

    # 创建.tif文件
    band_count = visual_ds.RasterCount  # 波段数
    xsize = visual_ds.RasterXSize
    ysize = visual_ds.RasterYSize
    out_tif_name = filename.split(".SAFE")[0] + ".tif"
    driver = gdal.GetDriverByName("GTiff")
    out_tif = driver.Create(out_tif_name, xsize, ysize, band_count, gdal.GDT_Float32)
    out_tif.SetProjection(visual_ds.GetProjection())  # 设置投影坐标
    out_tif.SetGeoTransform(visual_ds.GetGeoTransform())

    for index, band in enumerate(visual_arr):
        band = np.array([band])
        for i in range(len(band[:])):
            # 数据写出
            out_tif.GetRasterBand(index + 1).WriteArray(band[i])  # 将每个波段的数据写入内存，此时没有写入硬盘
    out_tif.FlushCache()  # 最终将数据写入硬盘
    out_tif = None  # 注意必须关闭tif文件

#创建图像金字塔
def build_pyramid(file_name):
    dataset = gdal.Open(file_name)
    dataset.BuildOverviews(overviewlist=[2, 4 ,8, 16])#查一下BuildOverviews参数设置含义
    del dataset

class processing_cut():
    def __init__(self,batch_size):
        batch_size = batch_size



    def image_read(self,image_path):
        # 从文件名的开头，判断是哪种图像
        filename = os.path.split(image_path)
        name = filename.split('_')[0]

        gdal.AllRegister()  # 先载入数据驱动，也就是初始化一个对象，让它“知道”某种数据结构，但是只能读，不能写
        ds = gdal.Open("d:/test/test.tif")  # 打开文件
        bands = ds.RasterCount()  # 获取波段数
        img_width, img_height = ds.RasterXSize, ds.RasterYSize  # 获取影像的宽高
        geotrans = ds.GetGeoTransform()  # 获取影像的投影信息

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
    #叶绿素
    ndvi = (nir - red) / (nir + red)

    #RI指数

    #辅助1指标

    #辅助2指标

    # NAN——>0
    nan_index = np.isnan(ndvi)
    ndvi[nan_index] = 0
    ndvi = ndvi.astype(np.float32)#对每个特征重复

    # 创建tif
    tif_driver = gdal.GetDriverByName('GTiff')
    out_ds = tif_driver.Create('features.tif', image_read.RasterXSize, image_read.RasterYSize, 4, gdal.GDT_Float32)
    # 设置投影坐标
    out_ds.SetProjection(image_read.GetProjection())
    out_ds.SetGeoTransform(image_read.GetGeoTransform())
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(ndvi)
    band1 = out_ds.GetRasterBand(1)
    band1.WriteArray(b1[0:5, 0:5] * 1.0 + 10)  # 原图是整数，乘个0.1，与浮点数可以相加 #修改
    band2 = out_ds.GetRasterBand(2)
    band2.WriteArray(b2[0:5, 0:5] * 1.0 + 20)
    band3 = out_ds.GetRasterBand(3)
    band3.WriteArray(b3[0:5, 0:5] * 1.0 + 30)
    band4 = out_ds.GetRasterBand(4)
    band4.WriteArray(b3[0:5, 0:5] * 1.0 + 40)
    return image_read = gdal.open(filename)

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
    #叶绿素
    ndvi = (nir - red) / (nir + red)

    #RI指数

    #辅助1指标

    #辅助2指标

    # NAN——>0
    nan_index = np.isnan(ndvi)
    ndvi[nan_index] = 0
    ndvi = ndvi.astype(np.float32)#对每个特征重复

    # 创建tif
    tif_driver = gdal.GetDriverByName('GTiff')
    out_ds = tif_driver.Create('features.tif', image_read.RasterXSize, image_read.RasterYSize, 4, gdal.GDT_Float32)
    # 设置投影坐标
    out_ds.SetProjection(image_read.GetProjection())
    out_ds.SetGeoTransform(image_read.GetGeoTransform())
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(ndvi)
    band1 = out_ds.GetRasterBand(1)
    band1.WriteArray(b1[0:5, 0:5] * 1.0 + 10)  # 原图是整数，乘个0.1，与浮点数可以相加 #修改
    band2 = out_ds.GetRasterBand(2)
    band2.WriteArray(b2[0:5, 0:5] * 1.0 + 20)
    band3 = out_ds.GetRasterBand(3)
    band3.WriteArray(b3[0:5, 0:5] * 1.0 + 30)
    band4 = out_ds.GetRasterBand(4)
    band4.WriteArray(b3[0:5, 0:5] * 1.0 + 40)
    return out_ds


def macro_feature(out_ds):#计算宏观指标
    image_read = gdal.open(out_ds)

    # 获取波段
    # b1波段
    b1 = image_read.GetRasterBand(1).ReadAsArray() * 0.0001
    # b2波段
    b2 = image_read.GetRasterBand(2).ReadAsArray() * 0.0001
    # b3波段
    b3 = image_read.GetRasterBand(3).ReadAsArray() * 0.0001
    # b4波段
    b4 = image_read.GetRasterBand(4).ReadAsArray() * 0.0001

    # 计算特征
    ndvi = (b1 - b2) / (b3 + b4)

    # NAN——>0
    nan_index = np.isnan(ndvi)
    ndvi[nan_index] = 0
    ndvi = ndvi.astype(np.float32)  # 对每个特征重复

    # 创建tif
    tif_driver = gdal.GetDriverByName('GTiff')
    out_ds_macro = tif_driver.Create('features_macro.tif', image_read.RasterXSize, image_read.RasterYSize, 1, gdal.GDT_Float32)
    # 设置投影坐标
    out_ds_macro.SetProjection(image_read.GetProjection())
    out_ds_macro.SetGeoTransform(image_read.GetGeoTransform())
    # 写入数据
    out_band = out_ds_macro.GetRasterBand(1)
    out_band.WriteArray(ndvi)
    return out_ds_macro

def cut_rsimage():
    pass


#找出排名前n的数据并找到对应位置
def topology_point_seek_band(downsize,out_ds,cut_size):
    #b1
    image_read = gdal.open(out_ds)
    #b1层找点
    b1_array = image_read.GetRasterBand(1).ReadAsArray(1)
    b1_cut = image_cut(b1_array,cut_size)
    point_b1 = topology_point_seek(b1_cut)








#找点
def topology_point_seek(downsize,point_count):
    for i in range(np.size(downsize,1)):
        for j in range(np.size(downsize, 0)):

            #取出排名前几的点



#把取出的点化成图，用dgl
def create_graph():









































    def preprocess(self,):
        #从文件名的开头，判断是哪种图像
        if self.name == 'S3A':


        if self.name=='':






