#!/usr/bin/env python3

import numpy as np
import sys
import os
import random
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from run_algo.algo_utils import define_algo
from run_algo.evaluation_metrics import cosine_metric, simple_regret, regret
from simulation_utils import get_feedback


 
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algo", type=str, default="DPB",
                    choices=['DPB', 'batch_active_PBL'], help="type of algorithm")
    ap.add_argument('-e', "--num-iteration", type=int, default=400,
                    help="# of iteration")
    ap.add_argument('-t', "--task-env", type=str, default="tosser",
                    help="type of simulation environment")
    ap.add_argument('-b', "--num-batch", type=int, default=10,
                    help="# of batch")
    ap.add_argument('-s' ,'--seed',  type=int, default=12345, help='A random seed')
    ap.add_argument('-w' ,'--exploration-weight',  type=float, default=0.5, help='DPB hyperparameter exploration weight')
    ap.add_argument('-g' ,'--discounting-factor',  type=float, default=0.99, help='DPB hyperparameter discounting factor')
    ap.add_argument('-d' ,'--delta',  type=float, default=0.7, help='DPB hyperparameter delta')
    ap.add_argument('-l' ,'--regularized-lambda',  type=float, default=1.0, help='DPB regularized lambda')
    ap.add_argument('-bm' ,'--BA-method',  type=str, default='random', choices=['greedy', 'medoids', 'dpp', 'random', 'information', 'max_regret'] ,help='method of batch active')
    
        
    args = vars(ap.parse_args())
    seed = args['seed']
    
    np.random.seed(seed)
    random.seed(seed)
    
    algos_type = args['algo']
    b = args['num_batch']
    N = args['num_iteration']
    task = args['task_env']
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    
    
    algo, true_w = define_algo(task, algos_type, args, 'simulated') # define algorithm, simulated human parameter
    
    
    t = 0
    t_th_w = 0 # changed parameter index
    
    
    turning_point = 30 # changes param per iteration
    
    # evaluation memory
    eval_cosine = [0]
    opt_simple_reward = [0]
    eval_simple_regret = [0]
    eval_cumulative_regret = [0]
    theta_hat = [[0,0,0,0]]

    
    while t < N:
        print('Samples so far: ' + str(t))

        if t!= 0 and t%(turning_point)==0:
            if t<290:
                t_th_w+=1
        
        algo.update_param(t)
        actions, inputA_set, inputB_set = algo.select_batch_actions(t, b)
        
        
        # evaluation 
        if t != 0:
            eval_cosine.append(cosine_metric(algo.hat_theta_D, true_w[t_th_w]))
            s_r, opt_reward = simple_regret(algo.predefined_features, algo.hat_theta_D, true_w[t_th_w])
            
            
            opt_simple_reward.append(opt_reward)
            eval_simple_regret.append(s_r)
            eval_cumulative_regret.append(eval_cumulative_regret[-1] + regret(algo.PSI , np.array(algo.action_s[-1:-b-1:-1]), true_w[t_th_w]))
            theta_hat.append(algo.hat_theta_D)
            
            
        #  (simulated) human feedback
        for i in range(b):
            
            featureA, featureB, A, R = get_feedback(algo, task, inputA_set[i], inputB_set[i],
                                actions[i], true_w[t_th_w], args, m="samling", human='simulated')
                
            if args['BA_method'] == 'information' or 'max_regret':
                algo.action_s.append(A)
                algo.action_s1.append(featureA) # save selected action
                algo.action_s2.append(featureB)
                algo.reward_s.append(R) # save label
            else :
                algo.action_s.append(A) # save selected action
                algo.reward_s.append(R) # save label
            
            
            t+=1
            
        if (t%15)==0:
            print(f"[{t}/{N}] Cos: {eval_cosine[-1]:.2f}, Sim Reg: {opt_simple_reward[-1] - eval_simple_regret[-1]:.2f}, Cum Reg: {eval_cumulative_regret[-1]:.2f}")
    