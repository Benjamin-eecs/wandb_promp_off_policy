from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder

import wandb
import random
import time

from meta_policy_search.meta_algos.pro_mp_off import ProMP_off
from meta_policy_search.meta_trainer_off import Trainer_off
from meta_policy_search.samplers.meta_sampler_off import MetaSampler_off
from meta_policy_search.samplers.meta_sample_processor_off import MetaSampleProcessor_off


import numpy as np
import tensorflow as tf
import os
import json




def main(tune_config = None):
    default_config = {
        'seed' : 0,


        # default config
        'num_tasks'                           : 2,
        'baseline'                            : 'LinearFeatureBaseline',
        'env'                                 : 'HalfCheetahRandDirecEnv',            
        # sampler config
        'rollouts_per_meta_task'              : 5,
        'max_path_length'                     : 100,
        'parallel'                            : True,
        # sample processor config
        'discount'                            : 0.99,
        'gae_lambda'                          : 1,
        'normalize_adv'                       : True,
        # policy config
        'hidden_sizes'                        : (64, 64),
        'learn_std'                           : True, # whether to learn the standard deviation of the gaussian policy
        # ProMP config
        'inner_lr'                            : 0.1, # adaptation step size
        'learning_rate'                       : 1e-4, # meta-policy gradient step size
        'num_promp_steps'                     : 5, # number of ProMp steps without re-sampling
        'clip_eps'                            : 0.3, # clipping range
        'target_inner_step'                   : 0.01,
        'init_inner_kl_penalty'               : 5e-4,
        'adaptive_inner_kl_penalty'           : False, # whether to use an adaptive or fixed KL-penalty coefficient
        'n_itr'                               : 1001, # number of overall training iterations
        'meta_batch_size'                     : 4, # number of sampled meta-tasks per iterations
        'num_inner_grad_steps'                : 1, # number of inner / adaptation gradient steps

    }


    test_Step_1_AverageReturn = np.zeros(default_config['n_itr'])        


    with wandb.init(config=tune_config):

        for seed in range(5):
            default_config.update(wandb.config)
            default_config.update({'seed':seed})
            print(default_config)


            set_seed(default_config['seed'])

            baseline =  globals()[default_config['baseline']]()    # instantiate baseline

            env = globals()[default_config['env']](default_config['seed']) # instantiate env

            env = normalize(env)                           # apply normalize wrapper to env
 
            policy          = MetaGaussianMLPPolicy(
                    name                  = "meta-policy",
                    obs_dim               = np.prod(env.observation_space.shape),
                    action_dim            = np.prod(env.action_space.shape),
                    meta_batch_size       = default_config['meta_batch_size'],
                    hidden_sizes          = default_config['hidden_sizes'],
                )

            sampler         = MetaSampler_off(
                env                       =  env,
                policy                    =  policy,
                rollouts_per_meta_task    =  default_config['rollouts_per_meta_task'],  # This batch_size is confusing
                meta_batch_size           =  default_config['meta_batch_size'],
                max_path_length           =  default_config['max_path_length'],
                parallel                  =  default_config['parallel'],
                buffer_length             =  default_config['buffer_length'],
                discount                  =  default_config['discount'],
                num_tasks                 =  default_config['num_tasks']
            )

            sample_processor = MetaSampleProcessor_off(
                baseline                  =  baseline,
                discount                  =  default_config['discount'],
                gae_lambda                =  default_config['gae_lambda'],
                normalize_adv             =  default_config['normalize_adv'],
            )

            algo             = ProMP_off(
                policy                    =  policy,
                inner_lr                  =  default_config['inner_lr'],
                meta_batch_size           =  default_config['meta_batch_size'],
                num_inner_grad_steps      =  default_config['num_inner_grad_steps'],
                learning_rate             =  default_config['learning_rate'],
                num_ppo_steps             =  default_config['num_promp_steps'],
                clip_eps                  =  default_config['clip_eps'],
                target_inner_step         =  default_config['target_inner_step'],
                init_inner_kl_penalty     =  default_config['init_inner_kl_penalty'],
                adaptive_inner_kl_penalty =  default_config['adaptive_inner_kl_penalty'],
                off_clip_eps_upper        =  default_config['off_clip_eps_upper'],
                off_clip_eps_lower        =  default_config['off_clip_eps_lower'],
                clip_style                =  default_config['clip_style']                 #0:TRPO 1:off policy pg

            )

            trainer = Trainer_off(
                algo                      =  algo,
                policy                    =  policy,
                env                       =  env,
                sampler                   =  sampler,
                sample_processor          =  sample_processor,
                n_itr                     =  default_config['n_itr'],
                seeds                     =  [default_config['seed']] * default_config['rollouts_per_meta_task'] * default_config['meta_batch_size'],
                num_inner_grad_steps      =  default_config['num_inner_grad_steps'],
                sample_batch_size         =  default_config['sample_batch_size'],
                
            )

            res                           = trainer.train()

            test_Step_1_AverageReturn     = test_Step_1_AverageReturn + np.array(res)

            del baseline
            del env
            del policy
            del sampler
            del sample_processor 
            del algo
            del trainer

        test_Step_1_AverageReturn = test_Step_1_AverageReturn / 5
        for itr in range(default_config['n_itr']):
            wandb.log({'test-Step_1-AverageReturn':test_Step_1_AverageReturn[itr]})



if __name__=="__main__":
    main(wandb.config)
