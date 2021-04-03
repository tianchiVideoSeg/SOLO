import os
import glob
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from time import time
from mmdet.apis import inference_detector, init_detector


def show_result_ins(img,
                    result,
                    score_thr=0.2,
                    sort_by_density=False,
                    out_file=None):
    """Visualize the instance segmentation results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The instance segmentation result.
        score_thr (float): The threshold to visualize the masks.
        sort_by_density (bool): sort the masks by their density.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """

    img = mmcv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape
    mask = np.zeros_like(img_show)

    if not result or result == [None]:
        return mask
    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    for idx in range(num_mask):
        idx = -(idx+1)
        cur_cate = cate_label[idx]
        if cur_cate == 0:
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            color_mask = color_masks[idx]
            cur_mask_bool = cur_mask.astype(np.bool)
            mask[cur_mask_bool] = color_mask

            cur_score = cate_score[idx]

    if out_file is None:
        return mask
    else:
        mmcv.imwrite(mask, out_file)


if __name__ == '__main__':
    # Choose to use a config and initialize the detector
    config = '../configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
    # Setup a checkpoint file to load
    checkpoint = '../checkpoints/SOLOv2_X101_DCN_3x.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')

    # video path settings
    root = '/home/baikai/Desktop/AliComp/datasets/PreRoundData/'
    video_sets = 'val.txt'
    video_dir = os.path.join(root, 'JPEGImages')
    mask_dir = os.path.join(root, 'MaskProps')
    txt_path = os.path.join(root, 'ImageSets', video_sets)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    folders = []
    f = open(txt_path, 'r')
    while True:
        x = f.readline()
        x = x.rstrip()
        if not x: break
        folders.append(os.path.join(video_dir, x))

    for i, video in enumerate(folders):
        if not os.path.exists(video.replace('JPEGImages', 'MaskProps')):
            os.makedirs(video.replace('JPEGImages', 'MaskProps'))
        frames = sorted(glob.glob(video + '/*'))
        mask_names = [frame.replace('JPEGImages', 'MaskProps').replace('jpg', 'png') for frame in frames]
        t = time()
        for frame, mask_name in zip(frames, mask_names):
            mask = inference_detector(model, frame)
            show_result_ins(frame, mask, score_thr=0.25, out_file=mask_name)
            del mask
        print('video', i, 'finished in', time() - t, 'seconds.', len(frames),
              'images at', (time() - t) / len(frames), 'seconds per image.')
