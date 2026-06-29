from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UISDataset(BaseSegDataset):
    METAINFO = {
        'classes': ('background', 'Slum'),  # 单分类：背景+目标
        'palette': [[0, 0, 0], [255, 255, 255]]  # 黑背景，白目标
    }

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)