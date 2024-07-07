import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


class PBL_model(object):
    def __init__(self, simulation_object, env='simulated'):
        
        self.simulation_object = simulation_object
        self.d = simulation_object.num_of_features
        
        data = np.load('../ctrl_samples/' + self.simulation_object.name + '.npz', allow_pickle=True)
        self.PSI = data['psi_set']
        if self.simulation_object.name == 'avoid':
            self.inputs_set = data['psi_set']
        else:
            self.inputs_set = data['inputs_set']
        features_data = np.load('../ctrl_samples/' + self.simulation_object.name + '_features'+'.npz', allow_pickle=True)
        self.predefined_features = features_data['features']
        
        self.action_s = [] # memory for selected actions
        self.action_s1 = []
        self.action_s2 = []
        self.reward_s = [] # memory for labeled
            
    
    def update_param(self): # parameter update rule
        raise NotImplementedError("must implement udate param method")
    def select_single_action(self): # single action selection rule
        raise NotImplementedError("must implement select single action method")
    def select_batch_actions(self): # batch actions selection rule
        raise NotImplementedError("must implement select single action method")
        
    
    
    
    
    
    
    
    
