import os
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class RumexWeedsDataset(BaseSegDataset):
    METAINFO = {
        'classes': ['background', 'rumex_obtusifolius', 'rumex_crispus'],
        'palette': [
            [0, 0, 0],    # background
            [0, 255, 0],  # rumex_obtusifolius
            [0, 0, 255],  # rumex_crispus
        ]
    }

    def __init__(self, data_root, img_dir, mask_dir, split_file=None, **kwargs):
        self.img_dir = os.path.join(data_root, img_dir)
        self.mask_dir = os.path.join(data_root, mask_dir)
        self.split_file = os.path.join(data_root, split_file) if split_file else None
        super().__init__(data_root=data_root, **kwargs)

    def load_data_list(self):
        if self.split_file and os.path.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                basenames = [os.path.splitext(os.path.basename(line.strip()))[0] for line in f if line.strip()]
        else:
            basenames = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith('.png')]
        data_list = []
        for name in basenames:
            img_path = os.path.join(self.img_dir, name + '.png')
            mask_path = os.path.join(self.mask_dir, name + '.png')
            data_list.append({
                'img_path': img_path,
                'seg_map_path': mask_path,
                'reduce_zero_label': False,
                'seg_fields': []
            })
        return data_list 