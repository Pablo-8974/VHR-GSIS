import torch

from mmseg.apis import inference_model, init_model, show_result_pyplot

from torchvision.utils import make_grid
import cv2
import os
import glob
import shutil
import numpy as np
import tifffile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
np.random.seed(2025)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def save_binary_method(model_pred, save_path):
    seg_image = model_pred.pred_sem_seg.data * 255

    seg_image = make_grid(seg_image).permute(1, 2, 0).to("cpu").numpy()
    cv2.imwrite(save_path, seg_image)


def save_logits_method(model_pred, save_path):
    seg_logits = model_pred.seg_logits.data.softmax(0)
    slum_logits = seg_logits[-1]
    seg_image = make_grid(slum_logits).permute(1, 2, 0).to("cpu").numpy()
    tifffile.imwrite(save_path, seg_image)


def inference_single_dir(data_path, save_path: str, model=None, save_path_added=True, save_logits=True):
    file_list = glob.glob(os.path.join(data_path, '*.png'))
    print('\nWork on', data_path)
    print('Save in', save_path)

    dir_name = os.path.basename(data_path)
    if save_path_added:
        _save_path = os.path.join(save_path, dir_name)
    else:
        _save_path = save_path
    os.makedirs(_save_path, exist_ok=True)

    if save_logits:
        save_func = save_logits_method
        print('Save Logits')
        save_suffix = 'tif'
    else:
        save_func = save_binary_method
        print('Save Binary')
        save_suffix = 'png'

    shutil.copy(os.path.join(data_path, 'split_log.txt'),
                os.path.join(_save_path, 'split_log.txt')
                )

    if model is None:
        model = init_model(config_file, checkpoint_file, device='cuda:0')
        print('Init model from ', config_file, '\n', 'load model from ', checkpoint_file)

    with torch.no_grad():
        for image_idx, image in enumerate(file_list):
            image_name = os.path.basename(image)
            image_name = image_name.replace('png', save_suffix)

            # image = cv2.imread(image)
            # reference_rgb = cv2.imread(os.path.join(color_histogram_reference_path, image_name))
            # 执行直方图匹配（RGB）
            # image = exposure.match_histograms(image, reference_rgb, channel_axis=-1)
            # image = image[:, :, [2, 1, 0]]   # to rgb

            result = inference_model(model, image)
            save_func(result, os.path.join(_save_path, image_name))

            print('\r', image_idx, '/', len(file_list), end='')



########################################################################################################################
#####################                      model setting, NO NEED TO CHANGE                        #####################
########################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
CUDA_VISIBLE_DEVICES="3"
config_file = '/intelnvme04/jiang.mingyu/slum/mmsegmentation-1.2.2/configs/segnext/Slum-test.py'
#
# # use which city's model  best_mIoU_epoch_0  Latest
# # Dhaka-best_mIoU_epoch_0 Mumbai-best_mIoU_epoch_0 CapeTown-Latest Manila-2500（old）-2200（new） Nairobi-3500
# # Caracas-3500 Rio_de_Janeiro-8700 Karachi-6800 Cairo-error-5400 Port-au-Prince-Mumbai     full-dataset
city_name = 'CapeTown'
checkpoint_file = f'/intelnvme04/jiang.mingyu/slum/mmsegmentation-1.2.2/slum-segnext/{city_name}/Latest.pth'
# color_histogram_reference_path = '/intelnvme04/jiang.mingyu/slum/GoogleMap/Caracas/2021/split'

########################################################################################################################
#####################                             inference_single_dir                             #####################
########################################################################################################################
inference_single_dir(
    '/intelnvme04/jiang.mingyu/slum/GoogleMap/Johannesburg/2010/split',
    '/intelnvme04/jiang.mingyu/slum/GoogleMap/Johannesburg/2010/inference',		# 注意这里是  **********************************************************
    save_path_added=False,  save_logits=False
)
print("Inference finished\n\n")

########################################################################################################################
#####################                            inference_single_image                            #####################
########################################################################################################################
# inference_image()

########################################################################################################################
#####################                              inference_list_dir                              #####################
########################################################################################################################
# inference_list_dir()
