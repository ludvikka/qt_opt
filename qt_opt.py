"""
An example of QT-Opt.
"""




import argparse
import copy
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import ArgmaxQfPol
from machina.noise import OUActionNoise
from machina.algos import qtopt
from machina.vfuncs import DeterministicSAVfunc, CEMDeterministicSAVfunc
from machina.envs import GymEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from simple_net import QNet

import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method("spawn")
    print("yeeeees")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='garbage',
                        help='Directory name of log.')
    parser.add_argument('--env_name', type=str,
                        default='MountainCarContinuous-v0', help='Name of environment.')
    parser.add_argument('--record', action='store_true',
                        default=False, help='If True, movie is saved.')
    parser.add_argument('--seed', type=int, default=256)
    parser.add_argument('--max_epis', type=int,
                        default=100000000, help='Number of episodes to run.')
    parser.add_argument('--max_steps_off', type=int,
                        default=1000000000000, help='Number of episodes stored in off traj.')
    parser.add_argument('--num_parallel', type=int, default=1,
                        help='Number of processes to sample.')
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')

    parser.add_argument('--max_steps_per_iter', type=int, default=4000,
                        help='Number of steps to use in an iteration.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pol_lr', type=float, default=1e-4,
                        help='Policy learning rate.')
    parser.add_argument('--qf_lr', type=float, default=1e-3,
                        help='Q function learning rate.')
    parser.add_argument('--h1', type=int, default=32,
                        help='hidden size of layer1.')
    parser.add_argument('--h2', type=int, default=32,
                        help='hidden size of layer2.')
    parser.add_argument('--tau', type=float, default=0.0001,
                        help='Coefficient of target function.')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor.')

    parser.add_argument('--lag', type=int, default=6000,
                        help='Lag of gradient steps of target function2.')
    parser.add_argument('--num_iter', type=int, default=2,
                        help='Number of iteration of CEM.')
    parser.add_argument('--num_sampling', type=int, default=60,
                        help='Number of samples sampled from Gaussian in CEM.')
    parser.add_argument('--num_best_sampling', type=int, default=6,
                        help='Number of best samples used for fitting Gaussian in CEM.')
    parser.add_argument('--multivari', action='store_true',
                        help='If true, Gaussian with diagonal covarince instead of Multivariate Gaussian matrix is used in CEM.')
    parser.add_argument('--eps', type=float, default=0.2,
                        help='Probability of random action in epsilon-greedy policy.')
    parser.add_argument('--loss_type', type=str,
                        choices=['mse', 'bce'], default='mse',
                        help='Choice for type of belleman loss.')
    parser.add_argument('--save_memory', action='store_true',
                        help='If true, save memory while need more computation time by for-sentence.')
    args = parser.parse_args()

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    with open(os.path.join(args.log, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    pprint(vars(args))

    if not os.path.exists(os.path.join(args.log, 'models')):
        os.makedirs(os.path.join(args.log, 'models'))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
    device = torch.device(device_name)
    set_device(device)
    print(device_name)

    score_file = os.path.join(args.log, 'progress.csv')
    logger.add_tabular_output(score_file)
    #logger.add_tensorboard_output(args.log)



    import gym
    import robosuite as suite


    from custom_task import LiftSquareObject

    from custom_gym_wrapper import CustomGymWrapper

    from robosuite import load_controller_config


    import numpy as np


    camera_quat = [0.6743090152740479, 0.21285612881183624, 0.21285581588745117, 0.6743084788322449]
    pos = [0.626,0,1.6815]
    height_vs_width_relattion = 754/449
    camera_attribs = {'fovy': 31.0350747}
    camera_h = 100
    camera_w = int(camera_h * height_vs_width_relattion)




    controller_config = load_controller_config(default_controller="JOINT_POSITION")
    env = suite.make(
        camera_pos = pos,#(1.1124,-0.046,1.615),#(1.341772827,  -0.312295471 ,  0.182150085+1.5), 
        camera_quat = camera_quat,#(0.5608417987823486, 0.4306466281414032, 0.4306466579437256, 0.5608419179916382),# frontview quat
        camera_attribs = camera_attribs,
        env_name="LiftSquareObject", # try with other tasks like "Stack" and "Door"
        robots="IIWA",  # try with other robots like "Sawyer" and "Jaco"
        gripper_types="Robotiq85Gripper",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names =['calibrated_camera'],
        camera_widths =[camera_w],
        camera_heights=[camera_h],
        camera_depths=[True],
        use_object_obs=False,
        controller_configs=controller_config,
        control_freq = 20,
        horizon = 20,
    )
        

    env = CustomGymWrapper(env,['calibrated_camera_image'])
    observation_space = env.observation_space
    action_space = env.action_space
    observation_space = observation_space.spaces['calibrated_camera_image']


    qf_net = QNet(observation_space, action_space, args.h1, args.h2)
    lagged_qf_net = QNet(observation_space, action_space, args.h1, args.h2)
    lagged_qf_net.load_state_dict(qf_net.state_dict())
    targ_qf1_net = QNet(observation_space, action_space, args.h1, args.h2)
    targ_qf1_net.load_state_dict(qf_net.state_dict())
    targ_qf2_net = QNet(observation_space, action_space, args.h1, args.h2)
    targ_qf2_net.load_state_dict(lagged_qf_net.state_dict())
    qf = DeterministicSAVfunc(observation_space, action_space, qf_net)
    lagged_qf = DeterministicSAVfunc(
        observation_space, action_space, lagged_qf_net)
    targ_qf1 = CEMDeterministicSAVfunc(observation_space, action_space, targ_qf1_net, num_sampling=args.num_sampling,
                                    num_best_sampling=args.num_best_sampling, num_iter=args.num_iter,
                                    multivari=args.multivari, save_memory=args.save_memory)
    targ_qf2 = DeterministicSAVfunc(
        observation_space, action_space, targ_qf2_net)

    pol = ArgmaxQfPol(observation_space, action_space, targ_qf1, eps=args.eps)

    sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)

    optim_qf = torch.optim.Adam(qf_net.parameters(), args.qf_lr)

    off_traj = Traj(args.max_steps_off, traj_device='cpu')

    total_epi = 0
    total_step = 0
    total_grad_step = 0
    num_update_lagged = 0
    max_rew = -1e6

    while args.max_epis > total_epi:
        print(total_epi)
        with measure('sample'):
            print("start collecting")
            epis = sampler.sample(pol, max_epis = 2 ,max_steps=2)
            print("done collecting")
        with measure('train'):
            print("start train")
            on_traj = Traj(traj_device='cpu')
            on_traj.add_epis(epis)

            on_traj = ef.add_next_obs(on_traj)
            on_traj.register_epis()

            off_traj.add_traj(on_traj)

            total_epi += on_traj.num_epi
            step = on_traj.num_step
            total_step += step
            epoch = step

            result_dict = qtopt.train(
                off_traj, qf, lagged_qf, targ_qf1, targ_qf2,
                optim_qf, epoch, args.batch_size,
                args.tau, args.gamma, loss_type=args.loss_type
            )
            print("done train")

        total_grad_step += epoch
        if total_grad_step >= args.lag * num_update_lagged:
            logger.log('Updated lagged qf!!')
            lagged_qf_net.load_state_dict(qf_net.state_dict())
            num_update_lagged += 1

        rewards = [np.sum(epi['rews']) for epi in epis]
        mean_rew = np.mean(rewards)
        logger.record_results(args.log, result_dict, score_file,
                            total_epi, step, total_step,
                            rewards,
                            plot_title=args.env_name)

        if mean_rew > max_rew:
            torch.save(pol.state_dict(), os.path.join(
                args.log, 'models', 'pol_max.pkl'))
            torch.save(qf.state_dict(), os.path.join(
                args.log, 'models',  'qf_max.pkl'))
            torch.save(targ_qf1.state_dict(), os.path.join(
                args.log, 'models',  'targ_qf1_max.pkl'))
            torch.save(targ_qf2.state_dict(), os.path.join(
                args.log, 'models',  'targ_qf2_max.pkl'))
            torch.save(optim_qf.state_dict(), os.path.join(
                args.log, 'models',  'optim_qf_max.pkl'))
            max_rew = mean_rew

        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models',  'pol_last.pkl'))
        torch.save(qf.state_dict(), os.path.join(
            args.log, 'models', 'qf_last.pkl'))
        torch.save(targ_qf1.state_dict(), os.path.join(
            args.log, 'models', 'targ_qf1_last.pkl'))
        torch.save(targ_qf2.state_dict(), os.path.join(
            args.log, 'models', 'targ_qf2_last.pkl'))
        torch.save(optim_qf.state_dict(), os.path.join(
            args.log, 'models',  'optim_qf_last.pkl'))
        del on_traj
    del sampler