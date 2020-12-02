import argparse
import logging
from torch._C import dtype

import torch.utils.data

import math
import gym
import peg_in_hole_gym
import pybullet as p
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.draw import polygon

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.dataset_processing.grasp import detect_grasps
from utils.data.grasp_data import collect_data, get_file_data

logging.basicConfig(level=logging.INFO)


# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

#     # Network
#     parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

#     # Dataset & Data & Training
#     parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
#     parser.add_argument('--dataset-path', type=str, help='Path to dataset')
#     parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
#     parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
#     parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
#     parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
#     parser.add_argument('--ds-rotate', type=float, default=0.0,
#                         help='Shift the start point of the dataset to use a different test/train split')
#     parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

#     parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
#     parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
#     parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
#     parser.add_argument('--vis', action='store_true', help='Visualise the network output')

#     args = parser.parse_args()

#     if args.jacquard_output and args.dataset != 'jacquard':
#         raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
#     if args.jacquard_output and args.augment:
#         raise ValueError('--jacquard-output can not be used with data augmentation.')

#     return args

network = 'epoch_00001_iou_1.6268' # path to model
use_rgb = True
use_depth = True
vis = True
iou_eval = True
max_size = 10
output_size = 300
n_grasps = 1
mp_num = 16
sub_num = 4
dataset_dir = './dataset/'

def get_gtbb(data, idx, rot=0, zoom=1.0):
    # gtbbs = grasp.GraspRectangles.load_from_pybullet_gym(idx, data, scale=output_size)# / 1024.0
    # c = output_size//2
    # # gtbbs.rotate(rot, (c, c))
    # # gtbbs.zoom(zoom, (c, c))
    # gtbbs.offset((output_size/2, output_size/2))
    
    x, y, angle, w, h = data[idx][4]
    a = (  w * np.sin(angle) + h * np.cos(angle) + 150.0, - w * np.cos(angle) + h * np.sin(angle) + 150.0)
    b = (  w * np.sin(angle) - h * np.cos(angle) - 150.0, + w * np.cos(angle) - h * np.sin(angle) + 150.0)
    c = (- w * np.sin(angle) - h * np.cos(angle) + 150.0, + w * np.cos(angle) + h * np.sin(angle) + 150.0)
    d = (- w * np.sin(angle) + h * np.cos(angle) - 150.0, - w * np.cos(angle) - h * np.sin(angle) + 150.0)
    gtbbs = {
             'a':a,
             'b':b,
             'c':c,
             'd':d,
             'angle':angle / 180. * math.pi
            }
    print("grs", gtbbs)
    return gtbbs


def iou_cal(gs1, gs2, angle_threshold=np.pi/6):
    if abs((gs1['angle'] - gs2['angle']  + np.pi/2) % np.pi - np.pi/2) > angle_threshold:
        return 0
    xx1 = [gs1['a'][0], gs1['b'][0], gs1['c'][0], gs1['d'][0]]
    yy1 = [gs1['a'][1], gs1['b'][1], gs1['c'][1], gs1['d'][1]]
    xx2 = [gs2['a'][0], gs2['b'][0], gs2['c'][0], gs2['d'][0]]
    yy2 = [gs2['a'][1], gs2['b'][1], gs2['c'][1], gs2['d'][1]]

    rr1, cc1 = polygon(xx1, yy1)
    rr2, cc2 = polygon(xx2, yy2)
    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0
    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    return intersection/union


if __name__ == '__main__':
    # args = parse_args()

    # Load Network
    net = torch.load(network)
    device = torch.device("cuda:0")

    
    # init env
    # env = gym.make('peg-in-hole-v0', client=p.DIRECT, task='peg-in-hole')
    # env.reset()
    # test_data = collect_data(env, max_size, mp_num, sub_num, 0)
    
    test_data = get_file_data(dataset_dir, max_size, 0)

    results = {'correct': 0, 'failed': 0}
    with torch.no_grad():
        for idx, (x, _,_,y,_ ) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            x = x.permute(2,0,1)
            x = x.unsqueeze(0)
            xc = x.to(device)
            yc = [yy.unsqueeze(0).to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)
            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])
            
            gs1 = detect_grasps(q_img, ang_img, width_img)
            detect_grasps(y[0].numpy().squeeze(), (torch.atan2(y[1], y[2]) / 2.0).numpy().squeeze(), y[3].numpy().squeeze())
            if gs1:
                
                points = gs1[0].as_gr.points
                gss1 = {
                        'a':points[0],
                        'b':points[1],
                        'c':points[2],
                        'd':points[3],
                        'angle':gs1[0].angle
                       }
                data2 = get_gtbb(test_data, idx)
                # gss2 = {
                #         'a':data2[0],
                #         'b':data2[1],
                #         'c':data2[2],
                #         'd':data2[3],
                #         'angle':test_data[idx][4][2]
                #        }
                gss2 = data2
                if iou_eval:
                    s = iou_cal(gss1, gss2)
                    if s > 0.25:
                        results['correct'] += 1
                    else:
                        results['failed'] += 1
            else:
                results['failed'] += 1
            # if args.jacquard_output:
            #     grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
            #     with open(jo_fn, 'a') as f:
            #         for g in grasps:
            #             f.write(test_data.dataset.get_jname(didx) + '\n')
            #             f.write(g.to_jacquard(scale=1024 / 300) + '\n')
            if vis:
                evaluation.plot_output(np.array(test_data[idx][0][:,:,1:4].numpy()[:,:,::-1], dtype=np.uint8),
                                       test_data[idx][0][:,:,0], q_img,
                                       ang_img, no_grasps=n_grasps, grasp_width_img=width_img)
                # gsgs = get_gtbb(test_data, idx)[0].points
                # img = Image.fromarray(np.array(test_data[idx][0][:,:,1:4].numpy()[:,:,::-1], dtype=np.uint8))
                # draw = ImageDraw.Draw(img)
                # draw.polygon([(i[0],i[1]) for i in gsgs], outline=25)
                # img.show()
                

    if iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))

    # if args.jacquard_output:
    #     logging.info('Jacquard output saved to {}'.format(jo_fn))

