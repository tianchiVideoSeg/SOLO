from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv


config_file = '../configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/SOLOv2_X101_DCN_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = '/home/baikai/Desktop/AliComp/datasets/PreRoundData/JPEGImages/628058/00001.jpg'
result = inference_detector(model, img)

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_out.jpg")
