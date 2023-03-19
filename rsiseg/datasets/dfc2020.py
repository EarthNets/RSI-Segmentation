# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DFC2020Dataset(CustomDataset):
    """DFC2020 dataset.
    
    We currently support Sentinel-2 13 bands. We use the official 5128/986 "test"/"validation" set as our "train/test" set.
    Please refer to RSI-Segmentation/tools/convert_datasets/dfc2020.py for data preparation.

    In segmentation map annotation for DFC2020, 0 stands for background, which
    is not included in 8 categories. ``reduce_zero_label`` is fixed to True.    

    """
    CLASSES = (
        'Forest', 'Shrubland', 'Grassland', 'Wetland', 'Cropland', 'Urban/Built-up', 'Barren', 'Water'
    )

    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]]


    def __init__(self, **kwargs):
        super(DFC2020Dataset, self).__init__(
            img_suffix='s2.tif',
            seg_map_suffix='lc.tif',
            reduce_zero_label=True,
            **kwargs)
