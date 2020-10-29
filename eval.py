import argparse
import logging
from torch._C import dtype

import torch.utils.data

import gym
import peg_in_hole_gym
import pybullet as p
import numpy as np

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
# from utils.data import data, get_dataset

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

network = 'output/epoch_00154_iou_0.0282' # path to model
use_rgb = True
use_depth = True
vis = True
iou_eval = True
max_size = 2
output_size = 300
n_grasps = 1

def get_gtbb(data, idx, rot=0, zoom=1.0):
    gtbbs = grasp.GraspRectangles.load_from_pybullet_gym(idx, data, scale=output_size)# / 1024.0
    c = output_size//2
    gtbbs.rotate(rot, (c, c))
    gtbbs.zoom(zoom, (c, c))
    return gtbbs

if __name__ == '__main__':
    # args = parse_args()

    # Load Network
    net = torch.load(network)
    device = torch.device("cuda:0")

    # # Load Dataset
    # logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    # Dataset = get_dataset(args.dataset)
    # test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
    #                        random_rotate=args.augment, random_zoom=args.augment,
    #                        include_depth=args.use_depth, include_rgb=args.use_rgb)
    # test_data = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.num_workers
    # )
    # logging.info('Done')
    
    # init env
    env = gym.make('peg-in-hole-v0', client=p.DIRECT, task='peg-in-hole')
    env.reset()
    test_data = []
    # print("ahaha")
    while len(test_data) < max_size:
        step_data = env.step([])
        # print([type(i) for i in step_data])
        # print(step_data[3])
        # print([torch.tensor(step_data[3][i]) for i in range(len(step_data[3]))])
        # print("attemp to collect",step_data[1], len(test_data))
        if step_data[1] != 0:
            step_data = list(step_data)
            step_data[0] = torch.from_numpy(step_data[0])
            # step_data[3] = [torch.tensor(step_data[3][i]) for i in range(len(step_data[3]))]
            # print("hoho", step_data[3])
            step_data.append(step_data[3][1])
            step_data[3] = [torch.from_numpy(np.expand_dims(s, 0).astype(np.float32)) for s in step_data[3][0]]
            test_data.append(step_data)
            logging.info('Collecting Data {:02d}/{}...'.format(len(test_data), max_size))
        env.reset() 
    results = {'correct': 0, 'failed': 0}

    # if args.jacquard_output:
    #     jo_fn = args.network + '_jacquard_output.txt'
    #     with open(jo_fn, 'w') as f:
    #         pass

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

            if iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, get_gtbb(test_data, idx),
                                                   no_grasps=n_grasps,
                                                   grasp_width=width_img,
                                                   )
                if s:
                    results['correct'] += 1
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

    if iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))

    # if args.jacquard_output:
    #     logging.info('Jacquard output saved to {}'.format(jo_fn))
