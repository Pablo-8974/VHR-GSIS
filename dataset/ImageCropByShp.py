from osgeo import gdal
import numpy as np


def shp_crop():
    # 打开栅格文件
    raster = gdal.Open(src_path)

    gdal.Warp(
        save_path,
        raster,
        format="GTiff",
        cutlineDSName=shp_path,
        cropToCutline=cropToCutline,
        dstNodata=0  # 设置裁剪区域外的像素值
    )


city_name = 'Mumbai'
year = 2022
src_path = f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/concat_mask.tif'
save_path = f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/concat_mask-crop.tif'
shp_path = f'/intelnvme04/jiang.mingyu/slum/GoogleMap/CityBoundary/India_Mumbai, MH_No.7_Level1.shp'
cropToCutline = False

shp_crop()
