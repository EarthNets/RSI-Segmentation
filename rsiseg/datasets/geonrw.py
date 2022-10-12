# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GeoNRWDataset(CustomDataset):
    """GeoNRW dataset.

    In segmentation map annotation for GeoNRW, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.    


    - prepare dataset: follow the instructions in https://github.com/EarthNets/Dataset4EO/blob/main/Dataset4EO/datasets/_builtin/geonrw.py
    - usage: 
        op1. load data with pytorch dataset (the current version): inheritate `CustomDataset`
        op2. load data with Dataset4EO datapipe: inheritate `EODataset`

    


    """
    CLASSES = (
        "forest",
        "water",
        "agricultural",
        "residential,commercial,industrial",
        "grassland,swamp,shrubbery",
        "railway,trainstation",
        "highway,squares",
        "airport,shipyard",
        "roads",
        "buildings"
    )

    PALETTE = [
        [44, 160, 44],
        [31, 119, 180],
        [140, 86, 75],
        [127, 127, 127],
        [188,189,34],
        [255,127,14],
        [148,103,189],
        [23,190,207],
        [214,39,40],
        [227, 119, 194]
        ]

    def __init__(self, **kwargs):
        super(GeoNRWDataset, self).__init__(
            img_suffix='rgb.jp2',
            seg_map_suffix='seg.tif',
            reduce_zero_label=True,
            **kwargs)
