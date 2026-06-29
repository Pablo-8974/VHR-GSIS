import shutil

from osgeo import gdal, ogr
import os
import glob
import numpy as np

import cv2
from PIL import Image

# src_path = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\Split_raster\32.tif'
# save_path = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\Crop_raster\32.tif'
# shapefile = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\label_shp\32.shp'


def crop_from_shp(src_path, shp_path, save_path):
    # 打开栅格文件
    raster = gdal.Open(src_path)

    # # 打开矢量文件
    # shp_ds = ogr.Open(shapefile)
    # shp_layer = shp_ds.GetLayer()

    gdal.Warp(
        save_path,
        raster,
        format="GTiff",
        cutlineDSName=shp_path,
        cropToCutline=False,
        dstNodata=0  # 设置裁剪区域外的像素值
    )

def make_negative_label(src_path, save_path):
    # 打开栅格文件
    raster = gdal.Open(src_path).ReadAsArray()

    out_label = np.zeros_like(raster)
    cv2.imwrite(save_path, out_label.transpose(1, 2, 0))

def color_image_with_label(color_img_path, label_img_path, out_path):
    # 读取彩色图像和标签图像
    color_img = Image.open(color_img_path).convert("RGBA")
    label_img = Image.open(label_img_path).convert("L")  # 灰度图像

    # 将标签图像转换为 NumPy 数组，并创建透明度遮罩
    label_array = np.array(label_img)
    mask = (label_array == 1).astype(np.uint8)  # 二值掩码

    # 创建一个红色半透明的叠加图层
    overlay = np.zeros((label_array.shape[0], label_array.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 255  # 红色通道
    overlay[..., 3] = mask * 96  # Alpha 通道（50% 透明度）

    # 转换为 PIL 图像
    overlay_img = Image.fromarray(overlay, mode="RGBA")

    # 合成图像
    result_img = Image.alpha_composite(color_img, overlay_img)

    # 保存结果
    result_img.save(out_path)

def color_shift_testing():
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    from PIL import Image

    image = cv2.imread(r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\dataset\images\testing\48.jpg')
    image = Image.fromarray(image)
    color_shift = T.ColorJitter(brightness=(0.5,0.9), contrast=(0.5,0.9), saturation=(0.4,0.9))(image)
    color_shift2 = T.ColorJitter(contrast=(0.5,0.51))(image)
    color_shift3 = T.ColorJitter(saturation=(0.5,0.51))(image)
    color_shift4 = T.ColorJitter(hue=0.2)(image)

    axs = plt.figure().subplots(1, 5)
    axs[0].imshow(image);axs[0].set_title('src');axs[0].axis('off')
    axs[1].imshow(color_shift);axs[1].set_title('brightness');axs[1].axis('off')
    axs[2].imshow(color_shift2);axs[2].set_title('contrast');axs[2].axis('off')
    axs[3].imshow(color_shift3);axs[3].set_title('saturation');axs[3].axis('off')
    axs[4].imshow(color_shift4);axs[4].set_title('hue');axs[4].axis('off')

    plt.show()

def label_three_channels_to_single_channel(data_path):
    # data_path = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Dhaka\L19\dataset\annotations\training'
    file_list = glob.glob(
        os.path.join(data_path, '*.png')
    )

    for sub_file_path in file_list:
        image = cv2.imread(sub_file_path)

        cv2.imwrite(sub_file_path, image[:, :, 0])
        print(sub_file_path)

# 颜色特征剔除实验
def to_minus_1_positive_1():
    image = cv2.imread(
        r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\dataset\images\3.png').transpose(2, 0, 1)

    # mean=[123.675, 116.28, 103.53]
    # std=[58.395, 57.12, 57.375]

    # 先进行数据标准化
    mean_1 = np.mean(image[0])
    mean_2 = np.mean(image[1])
    mean_3 = np.mean(image[2])

    var_1 = np.std(image[0])
    var_2 = np.std(image[1])
    var_3 = np.std(image[2])

    layer_1 = ((image[0] - mean_1) / var_1)[None, ...]
    layer_2 = ((image[1] - mean_2) / var_2)[None, ...]
    layer_3 = ((image[2] - mean_3) / var_3)[None, ...]

    out_mean_1 = np.mean(layer_1)
    out_mean_2 = np.mean(layer_2)
    out_mean_3 = np.mean(layer_3)

    out_var_1 = np.std(layer_1)
    out_var_2 = np.std(layer_2)
    out_var_3 = np.std(layer_3)

    # 再将其数值分布还原回0-1供检查
    min_value_1 = np.min(layer_1)
    min_value_2 = np.min(layer_2)
    min_value_3 = np.min(layer_3)

    # 重新回归正值
    layer_1 = layer_1 + abs(min_value_1)
    layer_2 = layer_2 + abs(min_value_2)
    layer_3 = layer_3 + abs(min_value_3)

    # 0-1
    max_value_1 = np.max(layer_1)
    max_value_2 = np.max(layer_2)
    max_value_3 = np.max(layer_3)

    layer_1 = layer_1 / max_value_1
    layer_2 = layer_2 / max_value_2
    layer_3 = layer_3 / max_value_3

    out_image = np.concatenate([layer_1, layer_2, layer_3], axis=0).transpose(1, 2, 0)
    cv2.imwrite('./normalization_test.png', out_image * 255)

def to_zero_positive_1():
    image = cv2.imread(
        r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\dataset\images\3.png').transpose(2, 0, 1)

    min_value_1 = np.min(image[0])
    min_value_2 = np.min(image[1])
    min_value_3 = np.min(image[2])

    max_value_1 = np.max(image[0])
    max_value_2 = np.max(image[1])
    max_value_3 = np.max(image[2])

    layer_1 = ((image[0] - min_value_1) / (max_value_1 + min_value_1))[None, ...]
    layer_2 = ((image[1] - min_value_2) / (max_value_2 + min_value_2))[None, ...]
    layer_3 = ((image[2] - min_value_3) / (max_value_3 + min_value_3))[None, ...]

    out_image = np.concatenate([layer_1, layer_2, layer_3], axis=0).transpose(1, 2, 0)
    # out_image = (out_image + 1) / 2

    cv2.imwrite('./normalization_test.png', out_image * 255)

def overlay_mask(color_image_path, mask_image_path):
    color_image = cv2.imread(color_image_path)
    mask_image = cv2.imread(mask_image_path)[:, :, 0]

    # if len(mask_image.shape) != 2:
    #     mask_image = mask_image[]

    red_channel = color_image[:, :, 2]
    height_light_area = mask_image > 0

    red_channel[height_light_area] = (red_channel[height_light_area] * 0.5 + mask_image[height_light_area] * 0.5).astype(np.uint8)

    color_image[:, :, 2] = red_channel
    return color_image

def overlay_mask_loop(image_dir, mask_dir, save_dir, image_suffix):
    os.makedirs(save_dir, exist_ok=True)

    file_list = glob.glob(os.path.join(image_dir, f'*.{image_suffix}'))
    total_files = len(file_list)
    for idx, sub_image_path in enumerate(file_list):
        image_name = os.path.basename(sub_image_path)
        sub_mask_path = os.path.join(mask_dir, image_name.replace(image_suffix, 'png'))      # .replace('jpg', 'png')

        overlay_image = overlay_mask(sub_image_path, sub_mask_path)
        cv2.imwrite(os.path.join(save_dir, image_name), overlay_image)

        print('\r', idx, '/', total_files, end='')
    shutil.copy(
        os.path.join(image_dir, 'split_log.txt'),
        os.path.join(save_dir, 'split_log.txt')
    )


if __name__ == '__main__':
    # crop_from_shp(
    #     r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\Split_raster\4.tif',
    #     r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\label_shp\4.shp',
    #     r'D:\learning\machine-learning\dataset\slum\GoogleMap\Mumbai\L19-25\Crop_raster\4.tif',
    # )

    # image_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Caracas\L19\dataset\images\training'
    # mask_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Caracas\L19\dataset\annotations\training'
    # save_dir = r'D:\learning\machine-learning\dataset\slum\GoogleMap\Caracas\L19\dataset' + r'\masked'
    # overlay_mask_loop(image_dir, mask_dir, save_dir, 'jpg')

    # # for concat information
    # shutil.copy(
    #     os.path.join(image_dir, 'split_log.txt'),
    #     os.path.join(save_dir, 'split_log.txt')
    # )

    # label_three_channels_to_single_channel(
    #     r'D:\learning\machine-learning\dataset\slum\GoogleMap\Manila\L19\dataset\annotations\training'
    # )

    # crop_from_shp(
    #     r'D:\learning\machine-learning\dataset\slum\GoogleMap\Karachi\L19\Split_raster\11.tif',
    #     r'D:\learning\machine-learning\dataset\slum\slum-label\karachi-2017\EO4SD_KARACHI_INFORMAL_2017.shp',
    #     r'D:\learning\machine-learning\dataset\slum\GoogleMap\Karachi\L19\Crop_raster\11.tif'
    # )
    pass
