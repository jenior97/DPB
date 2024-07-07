import numpy as np
import scipy.optimize as opt
import algos
import a_algos
from models import Driver, Tosser, Avoid
import os; import sys
from algorithms.DPB import DPB

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


def sampleBernoulli(mean):
    ''' 
    Function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1 
    else: return 0



def mu(x, theta):
    return 1/(1+np.exp(-np.dot(x, theta)))


def get_feedback(algo, task, input_A, input_B, psi, w, args, m ="samling", human='simulated'):
     
    s = 0

    if human=="simulated":
        while s==0:
            if task == 'avoid':
                if args['algo'] == 'batch_active_PBL':
                    # Max regret algorithm selects feature set idxs for its query selection rule
                    if args['BA_method'] == 'max_regret':
                        phi_A = algo.predefined_features[int(input_A)]
                        phi_B = algo.predefined_features[int(input_B)]
                    else:
                        phi_A = algo.predefined_features[int(input_A)]
                        phi_B = algo.predefined_features[int(input_A + len(algo.predefined_features)/2)]
                else:
                    phi_A = algo.predefined_features[int(input_A)]
                    phi_B = algo.predefined_features[int(input_A + len(algo.predefined_features)/2)]

            else :
                algo.simulation_object.feed(input_A)
                phi_A = algo.simulation_object.get_features()
                algo.simulation_object.feed(input_B)
                phi_B = algo.simulation_object.get_features()           
                
            if m == "samling":
                prefer_prob = mu(psi, w)
                s = sampleBernoulli(prefer_prob)
                if s == 0:
                    s=-1
  
            elif m == "oracle":
            
                # oracle model    
                if np.dot(psi, w)>0:
                    s = 1
                else:
                    s =-1

    if (type(algo) == DPB) and s == -1:
        s = 0
        
    return np.array(phi_A), np.array(phi_B), psi, s



def create_env(task):
    if task == 'driver':
        return Driver()
    elif task == 'tosser':
        return Tosser()
    elif task == 'avoid':
        return Avoid()
    else:
        print('There is no task called ' + task)
        exit(0)


def run_algo(method, simulation_object, w_samples, b=10, B=200, delta_samples=None, sample_logprobs=None):
    if simulation_object.name == "avoid":
        if method == 'nonbatch':
            return a_algos.nonbatch(simulation_object, w_samples)
        if method == 'greedy':
            return a_algos.greedy(simulation_object, w_samples, b)
        elif method == 'medoids':
            return a_algos.medoids(simulation_object, w_samples, b, B)
        elif method == 'boundary_medoids':
            return a_algos.boundary_medoids(simulation_object, w_samples, b, B)
        elif method == 'successive_elimination':
            return a_algos.successive_elimination(simulation_object, w_samples, b, B)
        elif method == 'random':
            return a_algos.random(simulation_object, w_samples, b)
        elif method == 'dpp':
            return a_algos.dpp(simulation_object, w_samples, b, B)
        elif method == 'information':
            return a_algos.information(simulation_object, w_samples, delta_samples, b)
        elif method == 'max_regret':
            return a_algos.max_regret(simulation_object, w_samples, sample_logprobs, b)
        else:
            print('There is no method called ' + method)
            exit(0)
        
    else:
        if method == 'nonbatch':
            return algos.nonbatch(simulation_object, w_samples)
        if method == 'greedy':
            return algos.greedy(simulation_object, w_samples, b)
        elif method == 'medoids':
            return algos.medoids(simulation_object, w_samples, b, B)
        elif method == 'boundary_medoids':
            return algos.boundary_medoids(simulation_object, w_samples, b, B)
        elif method == 'successive_elimination':
            return algos.successive_elimination(simulation_object, w_samples, b, B)
        elif method == 'random':
            return algos.random(simulation_object, w_samples, b)
        elif method == 'dpp':
            return algos.dpp(simulation_object, w_samples, b, B)
        elif method == 'information':
            return algos.information(simulation_object, w_samples, delta_samples, b)
        elif method == 'max_regret':
            return algos.max_regret(simulation_object, w_samples, sample_logprobs, b)
        else:
            print('There is no method called ' + method)
            exit(0)

