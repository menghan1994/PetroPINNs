from tqdm import tqdm 
from model.PINNModel import PINNModel
from CRM_PINN import load_all_training_data, read_test_data
# from analysis import load_PINN_model
from analysis_3D import load_PINN_model
import mat73
import numpy as np
import torch
import json

from gaft import GAEngine
from gaft.components import DecimalIndividual, Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import numpy as np
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

from pyswarm import pso
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = load_PINN_model()
model.to(device)


selected_time = np.concatenate((np.arange(1000), np.arange(1000, 3623, 5)))
# selected_time = np.arange(3623)

data_save_dir = './data/3D_Case/3D_Training_data_New/'
INJECTRATE = mat73.loadmat(os.path.join(data_save_dir, 'injrate.mat'))['sample']
OPR = mat73.loadmat(os.path.join(data_save_dir, 'OPR.mat'))['OPR'][selected_time]
WPR = mat73.loadmat(os.path.join(data_save_dir, 'WPR.mat'))['WPR'][selected_time]
TIME = mat73.loadmat(os.path.join(data_save_dir, 'Time.mat'))['Time'][selected_time]

TOTAL_WELL_NUMS = 30
INJECTOR_NUMS = 10
PRODUCER_NUMS = 20


time = TIME.reshape(-1, 1)
BHP = 500
train_producer_bottom_pressure = np.ones((time.shape[0], PRODUCER_NUMS)) * BHP
train_producer_bottom_pressure = torch.tensor(train_producer_bottom_pressure, dtype=torch.float, requires_grad=True)
train_T = torch.tensor(time, dtype=torch.float, requires_grad=True)

def input_data(*inject_rate):
    inject_rate = np.array([inject_rate])
    train_water_inject_rate = np.ones((time.shape[0], INJECTOR_NUMS)) * inject_rate
    train_water_inject_rate = torch.tensor(train_water_inject_rate, dtype=torch.float, requires_grad=True)
    return train_water_inject_rate, train_producer_bottom_pressure, train_T

def fitnessfunction(*inject_rate):
    train_water_inject_rate, train_producer_bottom_pressure, train_T = input_data(*inject_rate)
    train_water_inject_rate = train_water_inject_rate.to(device)
    train_producer_bottom_pressure = train_producer_bottom_pressure.to(device)
    train_T = train_T.to(device)

    output = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
    q_o = output[1]
    return q_o.sum().item()

# Define and register the fitness function
def fitness(indv):
    # inject_rate_1, inject_rate_2, inject_rate_3, inject_rate_4 = indv.solution
    injection_rate = indv.solution
    return fitnessfunction(*injection_rate)

def old_ga():
    def f(x):
        return -fitnessfunction(*x)
    print('start')
    varbound=np.array([[1000,40000]]*INJECTOR_NUMS)
    algorithm_param = {'max_num_iteration': 100,\
                   'population_size':200,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
    
    model=ga(function=f,dimension=INJECTOR_NUMS, variable_type='int',variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()


def main():
    # gaussian_Optimisation()
    for _ in range(10):
        old_ga()
    

if __name__ == '__main__':

    main()