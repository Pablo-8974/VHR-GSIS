import os
from osgeo import gdal
import cv2

import numpy as np
import tifffile

import glob
from osgeo import gdal, ogr, osr


def raster_to_shapefile(input_raster, output_shapefile, target_value=1):
    """将栅格中特定值(默认为1)的像素转换为shapefile"""
    # 打开输入栅格
    print("Raster to shp file")
    src_ds = gdal.Open(input_raster)
    if src_ds is None:
        raise ValueError("无法打开输入栅格文件")

    # 获取栅格信息
    band = src_ds.GetRasterBand(1)
    prj = src_ds.GetProjection()
    gt = src_ds.GetGeoTransform()

    # 读取数据为numpy数组
    data = band.ReadAsArray()

    # 创建内存中的临时shapefile
    driver = ogr.GetDriverByName("Memory")
    temp_ds = driver.CreateDataSource("temp")

    # 创建空间参考
    srs = osr.SpatialReference()
    srs.ImportFromWkt(prj)

    # 创建临时图层
    temp_layer = temp_ds.CreateLayer("polygons", srs, ogr.wkbPolygon)
    temp_layer.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))

    # 多边形化 - 修正这里传入图层而不是数据源
    gdal.Polygonize(band, None, temp_layer, 0, [], callback=None)

    # 创建最终输出的shapefile
    out_driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shapefile):
        out_driver.DeleteDataSource(output_shapefile)
    out_ds = out_driver.CreateDataSource(output_shapefile)
    out_layer = out_ds.CreateLayer("polygons", srs, ogr.wkbPolygon)
    out_layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))

    # 筛选目标值的要素并写入输出文件
    for feature in temp_layer:
        if feature.GetField("DN") == target_value:
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            out_feature.SetGeometry(feature.GetGeometryRef().Clone())
            out_feature.SetField("value", target_value)
            out_layer.CreateFeature(out_feature)
            out_feature = None

    # 关闭所有数据集
    temp_ds = None
    out_ds = None
    src_ds = None

    print(f"成功创建shapefile: {output_shapefile}")
    return output_shapefile


def save_tif(array, save_path, geotransform=None, projection=None):
    driver = gdal.GetDriverByName('GTiff')
    ndvi_ds = driver.Create(save_path, array.shape[1], array.shape[0], array.shape[2], gdal.GDT_Byte)		# Byte Float32------------------------------------------------------------

    for band in range(array.shape[2]):
        ndvi_ds.GetRasterBand(band + 1).WriteArray(array[:, :, band])

    if geotransform is not None:
        ndvi_ds.SetGeoTransform(geotransform)
    if projection is not None:
        ndvi_ds.SetProjection(projection)

    del ndvi_ds  # 写出文件后关闭文件


def split_data_concat(dir_path, patch_suffix):
    with open(os.path.join(dir_path, 'split_log.txt'), 'r') as f:
        log_file = f.readlines()
        num_h = int(log_file[0].split('\t')[1])
        num_w = int(log_file[1].split('\t')[1])
        c = int(log_file[2].split('\t')[1])
        h = int(log_file[3].split('\t')[1])
        w = int(log_file[4].split('\t')[1])
        patch_size = int(log_file[5].split('\t')[1])

        if len(log_file) > 6:
            overlap_size = int(log_file[6].split('\t')[1])
        else:
            overlap_size = 0
        patch_stride = patch_size - overlap_size

    total_patches = num_h * num_w
    zero_mask = np.zeros([h, w, 3], dtype=np.uint8)		# , dtype=np.uint8 --------------------------------------------------------------------------------------------------------------
    global_idx = 0
    for _h in range(num_h):
        h_start = _h * patch_stride
        h_end = (h_start + patch_size) if (_h + 1) != num_h else None

        for _w in range(num_w):
            w_start = _w * patch_stride
            w_end = (w_start + patch_size) if (_w + 1) != num_w else None

            patch = cv2.imread(os.path.join(dir_path, f'{global_idx}.{patch_suffix}'))
            # patch = tifffile.imread(os.path.join(dir_path, f'{global_idx}.{patch_suffix}'))
            zero_mask[h_start:h_end, w_start:w_end, :] += np.asarray(patch, dtype=np.uint8)     # 拼接预测分割图像用+=，彩色图像用=即可----------------------------------------------
            zero_mask[h_start:h_end, w_start:w_end, :] += patch     # 拼接预测分割图像用+=，彩色图像用=即可

            global_idx += 1
            print('\r', global_idx, '/', total_patches, end='')

    return zero_mask


def single_dir_split_data_concat(file_path, patch_suffix, origin_image_path, save_path):
    # file_path below is the .patch_suffix files
    # origin_image_path for projection and geotransform inform
    print('Concat data', file_path)
    save_type = os.path.basename(save_path).split('.')[-1]

    if save_type == 'tif':
        print('Save tif')
        origin_image = gdal.Open(origin_image_path)
        GeoTransform = origin_image.GetGeoTransform()
        Projection = origin_image.GetProjection()
        print('\n', GeoTransform)
        print(Projection, '\n')
        del origin_image

    concat_image = split_data_concat(
        file_path,
        patch_suffix
    )

    concat_image[concat_image > 0] = 255
    # concat_image = concat_image / 4.				#################################################################################
    save_type = os.path.basename(save_path).split('.')[-1]

    if save_type == 'tif':
        print('Save tif')
        origin_image = gdal.Open(origin_image_path)
        save_tif(
            concat_image, save_path,
            geotransform=GeoTransform,
            projection=Projection
        )

    else:
        print('Save png')
        cv2.imwrite(save_path, concat_image)


def overlap_crop(image_path, save_path, patch_size=256, overlap_size=128,
                 to_rgb=True, origin_suffix='tif', save_suffix='png'):
    patch_stride = patch_size - overlap_size     # 每次要移动多少个像素
    image_path = glob.glob(os.path.join(image_path, f'*.{origin_suffix}'))

    for sub_image_path in image_path:
        image_name = os.path.basename(sub_image_path)
        _save_path = os.path.join(save_path, 'split')

        os.makedirs(_save_path, exist_ok=True)

        print("Load Image from", sub_image_path)
        # image_array = tifffile.imread(sub_image_path).transpose(2, 0, 1)
        image_array = gdal.Open(sub_image_path).ReadAsArray()

        c, h, w = image_array.shape  # channel, height, width
        num_h = h // patch_stride
        num_w = w // patch_stride
        total_patches = num_h * num_w

        # output log
        log_file = open(os.path.join(_save_path, 'split_log.txt'), 'w+')
        # log_file = open(os.path.join("/intelnvme04/jiang.mingyu/slum/GoogleMap/z-not-for-use/SaoPaulo/2025", 'split_log.txt'), 'w+')
        log_file.write(f'num_h\t{num_h}\nnum_w\t{num_w}\n'
                       f'channel\t{c}\nheight\t{h}\nwidth\t{w}\n'
                       f'patch_size\t{patch_size}\n'
                       f'overlap_size\t{overlap_size}')
        log_file.close()
        print('Log finished!')

        global_idx = 0

        if to_rgb:
            image_array = image_array[[2, 1, 0]]

        for _h in range(num_h):
            h_start = _h * patch_stride
            h_end = (h_start + patch_size) if (_h + 1) != num_h else None

            for _w in range(num_w):
                w_start = _w * patch_stride
                w_end = (w_start + patch_size) if (_w + 1) != num_w else None

                patch = image_array[:, h_start:h_end, w_start:w_end].transpose(1, 2, 0)
                cv2.imwrite(
                    os.path.join(_save_path, f'{global_idx}.{save_suffix}'),
                    patch
                )
                print('\r', global_idx, '/', total_patches, end='')
                global_idx += 1

        print(f'file {image_name} process finished, \n'
              f'obtain {num_h} * {num_w} = {num_h * num_w} patches in total, \n'
              f'save to {_save_path}\n\n')
        break


def overwrite_projection():
    # 读取原始图像获取投影信息，随后删除节约内存
    print('Save tif')
    origin_image = gdal.Open(origin_image_path)
    GeoTransform = origin_image.GetGeoTransform()
    Projection = origin_image.GetProjection()
    print('\n', GeoTransform)
    print(Projection, '\n')
    del origin_image

    # 读png图像，raster信息
    concat_image = gdal.Open(concat_image_path).ReadAsArray()
    concat_image[concat_image > 0] = 255

    # 保存tif图像，并覆写投影信息
    print('Save tif')
    save_tif(
        concat_image, save_path,
        geotransform=GeoTransform,
        projection=Projection
    )



if __name__ == '__main__':
    #------------------------------------------------------------------------------------------------------------------#
    # 重写文件投影
    if 0:
        origin_image_path = '/intelnvme04/jiang.mingyu/slum/GoogleMap/Delhi/2010/L19.tif'
        concat_image_path = '/intelnvme04/jiang.mingyu/slum/GoogleMap/Delhi/2010/concat_mask.png'

        upper_path = os.path.dirname(origin_image_path)
        save_path = os.path.join(upper_path, 'concat_mask.tif')
        shp_mask_save_path = os.path.join(upper_path, 'mask-2010.shp')               # f'mask-{year}.shp'

        overwrite_projection()

        raster_to_shapefile(
            save_path, shp_mask_save_path, 255
        )

    #------------------------------------------------------------------------------------------------------------------#
    # One step for concat inference     Output tif file
    if 1:
        city_name = 'Johannesburg'		# z-not-for-use/SaoPaulo  Metro
        year = 2010
        tiff_save_name = 'concat_mask.tif'
        shp_mask_save_name = f'CapeTown-{year}.shp'                     # f'mask-{year}.shp'

        single_dir_split_data_concat(
            f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/inference-png',  # 注意这里是 文件夹代号 是 还是2 ********************************************
            'png',					# 注意这里是 文件类型 ********************************************
            f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/L19.tif',
            f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/{tiff_save_name}', # png, tif  concat_mask
        )

        if 1:
            raster_to_shapefile(
                f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/{tiff_save_name}',
                f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/{shp_mask_save_name}', 255
            )

    print('\n')

    #------------------------------------------------------------------------------------------------------------------#
    # For data split for testing data
    # no overlap crop
    # GF_data_split()
    if 0:
        overlap_crop(
            '/intelnvme04/jiang.mingyu/slum/GoogleMap//Johannesburg/2010',		# 2025  2010
            '/intelnvme04/jiang.mingyu/slum/GoogleMap//Johannesburg/2010',
            patch_size=416, overlap_size=208
        )

    #------------------------------------------------------------------------------------------------------------------#
    # tif_to_png(
    #     r'D:\learning\machine-learning\dataset\slum\GoogleMap\Rio_de_Janeiro\test.tif',
    #     r'D:\learning\machine-learning\dataset\slum\GoogleMap\Rio_de_Janeiro\test.png',
    #     to_rgb=True
    # )

    if 0:
        city_name = 'Delhi'
        year = 2010
        raster_to_shapefile(
            f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/concat_mask.png',
            f'/intelnvme04/jiang.mingyu/slum/GoogleMap/{city_name}/{year}/mask-{year}.shp', 255
        )

    pass
