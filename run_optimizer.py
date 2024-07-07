from simulation_utils import create_env
import sys
import numpy as np
from tqdm import trange


def get_abs_opt_f(predefined_features, w):
    
    opt_feature_id = np.argmax(np.abs(np.dot(w, predefined_features.T)))
    
    return predefined_features[opt_feature_id]

def get_opt_f(predefined_features, w):

    opt_feature_id = np.argmax(np.dot(w, predefined_features.T))
    
    return predefined_features[opt_feature_id]

def get_opt_id(predefined_features, w):
    
    opt_feature_id = np.argmax(np.dot(w, predefined_features.T))
    
    return opt_feature_id