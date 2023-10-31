
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from ..data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES


def get_class_names(dataset_name: str):
    # COCO panoptic
    if dataset_name == "coco_2017_train_panoptic" or \
        dataset_name == "coco_2017_val_panoptic_with_sem_seg":
        class_names = [x['name'] for x in COCO_CATEGORIES]
    # ADE 150
    elif dataset_name == "ade20k_panoptic_val" or \
        dataset_name == "ade20k_panoptic_train":
        class_names = [x['name'] for x in ADE20K_150_CATEGORIES]
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")

    if 'train' in dataset_name:
        class_names.append('other')
    return class_names
