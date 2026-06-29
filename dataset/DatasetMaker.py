import os
import shutil

from osgeo import gdal
import cv2

import numpy as np

import glob

from utils import make_negative_label, crop_from_shp, color_image_with_label, label_three_channels_to_single_channel


def save_tif(array, save_path, geotransform=None, projection=None):
    driver = gdal.GetDriverByName('GTiff')
    ndvi_ds = driver.Create(save_path, array.shape[1], array.shape[0], array.shape[2], gdal.GDT_Byte)

    for band in range(array.shape[2]):
        ndvi_ds.GetRasterBand(band + 1).WriteArray(array[:, :, band])

    if geotransform is not None:
        ndvi_ds.SetGeoTransform(geotransform)
    if projection is not None:
        ndvi_ds.SetProjection(projection)

    del ndvi_ds  # 写出文件后关闭文件


def GF_data_split():
    data_path = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Karachi\L19\cropped_label'        # 底下就是图片 Split_raster cropped_label
    save_path = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Karachi\L19\dataset\annotations' + r'\training'   # 保存路径 images annotations
    patch_size = 416                                              # 保存路径 images annotations training testing

    os.makedirs(save_path, exist_ok=True)

    if os.path.basename(data_path) == 'cropped_label':
        origin_suffix = 'png'
        save_suffix = 'png'
    else:
        origin_suffix = 'tif'
        save_suffix = 'jpg'

    # 批量化处理
    data_list = os.listdir(data_path)
    data_list.sort(key=lambda x: int(x.split('.')[0]))

    start_idx_from = 0
    global_idx = 0
    for image_name in data_list:
        image_base_name = image_name.split('.')
        if image_base_name[-1] != origin_suffix or len(image_base_name) == 1:
            continue

        image_base_name, suffix = image_base_name

        ##############################################
        #                   Filter                   #
        ##############################################
        # if image_base_name in ['1', '3', '4', '5', '7', '8', '9', '10', '13']:
        #     continue

        patch_save_path = save_path

        image_array = gdal.Open(
            os.path.join(data_path, image_name)
        ).ReadAsArray()

        c, h, w = image_array.shape     # channel, height, width
        num_h = h // patch_size
        num_w = w // patch_size

        image_array = image_array[[2, 1, 0]]
        for _h in range(num_h):
            h_start = _h * patch_size
            h_end = (_h + 1) * patch_size if (_h + 1) != num_h else -1

            for _w in range(num_w):
                w_start = _w * patch_size
                w_end = (_w + 1) * patch_size if (_w + 1) != num_w else -1

                if (_w + 1) != num_w:
                    patch = image_array[:, h_start:h_end, w_start:w_end].transpose(1, 2, 0)
                else:
                    patch = image_array[:, h_start:h_end, w_start:].transpose(1, 2, 0)

                # if not np.sum(patch):   # 图像全为0时排除
                #     continue

                cv2.imwrite(
                    os.path.join(patch_save_path, f'{global_idx}.{save_suffix}'),
                    patch
                )

                global_idx += 1

        # output log
        log_file = open(os.path.join(patch_save_path, f'split_log.txt'), 'a')
        log_file.write(f'num_h\t{num_h}\nnum_w\t{num_w}\n'
                       f'channel\t{c}\nheight\t{h}\nwidth\t{w}\n'
                       f'patch_size\t{patch_size}\n'
                       f'{image_base_name} from {start_idx_from} to {global_idx-1}\n\n')
        log_file.close()

        print(f'file {image_name} process finished, from {start_idx_from} to {global_idx-1}\n'
              f'obtain {num_h} * {num_w} = {num_h*num_w} patches in total, \n'
              f'save to {patch_save_path}\n\n')
        start_idx_from = global_idx


def single_dir_split_data_concat(dir_path, patch_suffix):
    # dir_path = r'D:\learning\machine-learning\dataset\slum\Cities\model_out\H48F015016'
    # concat_save_path = r'D:\learning\machine-learning\dataset\slum\Cities\model_out'

    # patch_suffix = 'png'
    # concat_save_suffix = 'png'

    # image_name = os.path.basename(dir_path)

    with open(os.path.join(dir_path, 'split_log.txt'), 'r') as f:
        log_file = f.readlines()
        num_h = int(log_file[0].split('\t')[1])
        num_w = int(log_file[1].split('\t')[1])
        c = int(log_file[2].split('\t')[1])
        h = int(log_file[3].split('\t')[1])
        w = int(log_file[4].split('\t')[1])
        patch_size = int(log_file[5].split('\t')[1])

    zero_mask = np.zeros([h, w, 3])

    global_idx = 0
    for _h in range(num_h):
        h_start = _h * patch_size
        h_end = (_h + 1) * patch_size if (_h + 1) != num_h else -1

        for _w in range(num_w):
            w_start = _w * patch_size
            w_end = (_w + 1) * patch_size if (_w + 1) != num_w else -1

            patch = cv2.imread(os.path.join(dir_path, f'{global_idx}.{patch_suffix}'))

            if (_w + 1) != num_w:
                zero_mask[h_start:h_end, w_start:w_end, :] = patch
            else:
                zero_mask[h_start:h_end, w_start:, :] = patch

            global_idx += 1
            print('\r', global_idx, '/', num_h * num_w, end='')

    return zero_mask

    # cv2.imwrite(
    #     os.path.join(concat_save_path, f'{image_name}.{concat_save_suffix}'),
    #     zero_mask
    # )


def list_dir_split_data_concat():
    list_dir_path = r'C:\Users\k\Desktop\slum\test_data\inference'
    save_path = r'C:\Users\k\Desktop\slum\test_data\inference'
    origin_image_path = r'C:\Users\k\Desktop\slum\test_data'

    patch_suffix = 'png'
    concat_save_suffix = 'tif'

    dir_list = os.listdir(list_dir_path)
    for idx, dir_name in enumerate(dir_list):

        ##############################################
        #                   Filter                   #
        ##############################################
        # if dir_name == '1':
        #     continue
        #
        # if len(dir_name.split('.')) != 1:
        #     continue

        concat_image = single_dir_split_data_concat(
            os.path.join(list_dir_path, dir_name),
            patch_suffix
        )
        # concat_image = concat_image.astype(np.int8)

        origin_image = gdal.Open(
            os.path.join(origin_image_path, f'{dir_name}.tif')
        )

        save_tif(
            concat_image, os.path.join(save_path, f'{dir_name}.{concat_save_suffix}'),
            geotransform=origin_image.GetGeoTransform(), projection=origin_image.GetProjection()
        )

        print('\n', dir_name, 'finished', idx, '/', len(dir_list))


def crop_image_loop():
    # src_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\CapeTown\L19\Split_raster'
    # shp_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\CapeTown\L19\label_shp'
    # out_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\CapeTown\L19\Crop_raster'

    src_dir = base_path + r'\Split_raster'
    shp_dir = base_path + r'\label_shp'
    out_dir = base_path + r'\Crop_raster'

    # One shp for all images
    # fix_shp_file = None
    fix_shp_file = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Karachi\slum.shp'

    os.makedirs(out_dir, exist_ok=True)

    image_path_list = glob.glob(
        os.path.join(src_dir, '*.tif')
    )

    for sub_image_path in image_path_list:
        image_name = os.path.basename(sub_image_path)

        if fix_shp_file is None:
            sub_shp_path = os.path.join(shp_dir, image_name.replace('tif', 'shp'))
        else:
            sub_shp_path = fix_shp_file

        if not os.path.exists(sub_shp_path):
            print(f'shp file {sub_shp_path} not exist, treat as negative sample')
            make_negative_label(sub_image_path, os.path.join(out_dir, image_name))
            continue

        crop_from_shp(
            sub_image_path, sub_shp_path,
            os.path.join(out_dir, image_name)
        )

        print(image_name)

def cropped_image_to_label():
    # cropped_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\CapeTown\L19\Crop_raster'
    # out_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\CapeTown\L19\cropped_label'

    cropped_dir = base_path + r'\Crop_raster'
    out_dir = base_path + r'\cropped_label'

    os.makedirs(out_dir, exist_ok=True)

    image_path_list = glob.glob(
        os.path.join(cropped_dir, '*.tif')
    )

    for sub_image_path in image_path_list:
        image_name = os.path.basename(sub_image_path)

        image_arr = gdal.Open(sub_image_path).ReadAsArray()
        out_label = np.sum(image_arr, axis=0)[None, :]

        mask = out_label == 0
        out_label[~mask] = 1
        out_label = np.concatenate([out_label, out_label, out_label], axis=0).transpose(1, 2, 0).astype(np.int8)

        cv2.imwrite(
            os.path.join(out_dir, image_name.replace('tif', 'png')), out_label
        )

        print(image_name)

def visualize_dataset():
    # image_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\CapeTown\L19\Split_raster'
    # label_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\CapeTown\L19\cropped_label'
    # out_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\CapeTown\L19\colored_label'

    image_dir = base_path + r'\Split_raster'
    label_dir = base_path + r'\cropped_label'
    out_dir = base_path + r'\colored_label'

    image_suffix = 'tif'

    os.makedirs(out_dir,exist_ok=True)
    image_path_list = glob.glob(
        os.path.join(image_dir, '*.' + image_suffix)
    )

    for sub_image_path in image_path_list:
        image_name = os.path.basename(sub_image_path)

        color_image_with_label(
            sub_image_path, os.path.join(label_dir, image_name.replace(image_suffix, 'png')),
            os.path.join(out_dir, image_name.replace(image_suffix, 'png'))
        )
        print(image_name)


def copy_file_loop(source_path, to_path, suffix='png'):
    copy_file_list = glob.glob(os.path.join(source_path, '*.' + suffix))
    start_name = len(glob.glob(os.path.join(to_path, '*.' + suffix)))

    for sub_image_path in copy_file_list:
        shutil.copy(
            sub_image_path, os.path.join(to_path, f'{start_name}.png')
        )

        start_name += 1

# list_dir_split_data_concat()

base_path = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Karachi\L19'

# step 1 制作大图标签
# crop_image_loop()       # 将图像按照shp文件裁切
# print('Crop image to positive area finished!')
# cropped_image_to_label()    # 使用裁切好的图像制作标签
# print('Making bool label finished!')
# visualize_dataset()     # 给原始彩色图像赋予红色透明度为50%的mask遮罩，标签可视化
# print('Mask label to optical image finished!')

# step 2 数据划分，切成小patch
# GF_data_split()         # 切割大图像
# print("Dataset making finished!")

# step 3 将标签通道由三个变为一个
# from utils import label_three_channels_to_single_channel
# label_three_channels_to_single_channel(
#         base_path + r'\dataset\annotations\testing'        # training testing annotations images
#     )

########################################################################################################################
#####################                        copy and rename image from dir                        #####################
########################################################################################################################
copy_file_loop(
    r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\dataset\annotations\training',
    r'D:\learning\machine-learning\dataset\slum\GoogleMap\Karachi\L19\dataset\annotations\training',
    suffix='png'
)
