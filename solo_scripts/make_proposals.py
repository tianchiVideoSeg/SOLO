import os
import glob
import mmcv
import cv2
import json
import numpy as np
import pycocotools.mask as maskUtils
from time import time
from mmdet.apis import inference_detector, init_detector


def read_flo(flo_name):
    with open(flo_name, 'rb') as f:
        magic, = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w, h = np.fromfile(f, np.int32, count=2)
            # print(f'Reading {w} x {h} flo file')
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (bands, columns, rows)
            flo = np.resize(data, (h, w, 2))

    return flo


def warp_flow(img, flow, binarize=True):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    if binarize:
        res = np.equal(res, 1).astype(np.uint8)
    return res


def save_result(result,
                flo,
                score_thr=0.2,
                out_file=None):
    """Visualize the instance segmentation results on the image.

    Args:
        result (tuple[list] or list): The instance segmentation result.
        score_thr (float): The threshold to visualize the masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
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

    # save segmentation masks
    props = []
    for idx in range(num_mask):
        idx = -(idx+1)
        cur_cate = cate_label[idx]
        # show result only when class(i) belongs to person
        if cur_cate == 0:
            score = cate_score[idx]
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (cur_mask.shape[1], cur_mask.shape[0]))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            seg = maskUtils.encode(np.asfortranarray(cur_mask))
            seg['counts'] = seg['counts'].decode()
            f_seg = None
            if flo is not None:
                warp_mask = warp_flow(cur_mask, flo)
                f_seg = maskUtils.encode(np.asfortranarray(warp_mask))
                f_seg['counts'] = f_seg['counts'].decode()
            props.append({"score": float(score),
                          "segmentation": seg,
                          "forward_segmentation": f_seg,
                          "backward_segmentation": None})
    # if out_file specified, do not show image in window
    if out_file is not None:
        # result will be written to the out file
        with open(out_file, 'w') as f:
            json.dump(props, f)


if __name__ == '__main__':
    # Choose to use a config and initialize the detector
    config = '../configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
    # Setup a checkpoint file to load
    checkpoint = '../solov2_x101_dcn_fpn_8gpu_3x/epoch_37.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')

    # video path settings
    root = '/home/baikai/Desktop/AliComp/datasets/PreRoundData/'
    video_sets = 'test.txt'
    video_dir = os.path.join(root, 'JPEGImages')
    prop_dir = os.path.join(root, 'proposals')
    txt_path = os.path.join(root, 'ImageSets', video_sets)
    if not os.path.exists(prop_dir):
        os.makedirs(prop_dir)

    folders = []
    f = open(txt_path, 'r')
    while True:
        x = f.readline()
        x = x.rstrip()
        if not x: break
        folders.append(os.path.join(video_dir, x))

    for i, video in enumerate(folders):
        if not os.path.exists(video.replace('JPEGImages', 'proposals')):
            os.makedirs(video.replace('JPEGImages', 'proposals'))
        frames = sorted(glob.glob(video + '/*'))
        prop_names = [frame.replace('JPEGImages', 'proposals').replace('jpg', 'json') for frame in frames]
        flo_names = [frame.replace('JPEGImages', 'Flows').replace('jpg', 'flo') for frame in frames]
        t = time()
        for frame, prop_name, flo_name in zip(frames[:-1], prop_names[:-1], flo_names[:-1]):
            mask = inference_detector(model, frame)
            flo = read_flo(flo_name)
            save_result(mask, flo, score_thr=0.3, out_file=prop_name)
            del mask
        mask = inference_detector(model, frames[-1])
        save_result(mask, None, score_thr=0.3, out_file=prop_names[-1])
        del mask
        print('video', i, 'finished in', time() - t, 'seconds.', len(frames),
              'images at', (time() - t) / len(frames), 'seconds per image.')
