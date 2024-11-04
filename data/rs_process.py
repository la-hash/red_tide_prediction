#实现以下几个函数功能
#裁剪（裁剪前先画shapefile文件）
#辐射矫正
import os
import gdal





def h5_to_tiff():#或者用ENVI导出


def image_cut():
    input_file = 'G:/Test/'
    output_file = 'G:/Test/Clip/'
    input_shape = 'G:/Test/Test.shp'
    prefix = '.tif'

    if not os.path.exists(output_file):
        os.mkdir(output_file)

    file_all = os.listdir(input_file)
    for file_i in file_all:
        if file_i.endswith(prefix):
            file_name = input_file + file_i
            data = gdal.Open(file_name)

            ds = gdal.Warp(output_file + os.path.splitext(file_i)[0] + '_Clip.tif',
                           data,
                           format='GTiff',
                           cutlineDSName=input_shape,
                           cutlineWhere="FIELD = 'whatever'",
                           dstNodata=0)

def radiation_correction():#也可以用ENVI进行辐射校正，先校正，再转存格式


