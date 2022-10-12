from .builder import DATASETS
from .custom import EODataset


@DATASETS.register_module()
class VaihingenDataset(EODataset):

    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]

    #def __init__(self, **kwargs):
    #    super(VaihingenDataset, self).__init__(
    #        reduce_zero_label=True,
    #        **kwargs)