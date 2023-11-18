# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import mat73
import numpy as np
import json
from tqdm import tqdm 
from model.PINNModel import PINNModel
from model.PureMachineLearning import PureMachineLearning


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def read_training_data(case_ind):
    inject_rate = INJECTRATE[case_ind:case_ind+1, :]
    slice_index = np.arange(case_ind*TOTAL_WELL_NUMS, (case_ind+1)*TOTAL_WELL_NUMS)
    opr = OPR[:, slice_index][:, INJECTOR_NUMS:]
    wpr = WPR[:, slice_index][:, INJECTOR_NUMS:]
    time = TIME.reshape(-1, 1)
    return inject_rate, opr, wpr, time

def load_training_data(case_ind):
    injection_rate, opr, wpr, time = read_training_data(case_ind)
    BHP = 500
    train_producer_bottom_pressure = np.ones((opr.shape[0], PRODUCER_NUMS)) * BHP
    train_water_inject_rate = np.ones((opr.shape[0], INJECTOR_NUMS)) * injection_rate
    train_producer_bottom_pressure = torch.tensor(train_producer_bottom_pressure, dtype=torch.float, requires_grad=True)
    train_water_inject_rate = torch.tensor(train_water_inject_rate, dtype=torch.float, requires_grad=True)
    train_T = torch.tensor(time, dtype=torch.float, requires_grad=True)
    train_OPR = torch.tensor(opr, dtype=torch.float, requires_grad=True)
    train_WPR = torch.tensor(wpr, dtype=torch.float, requires_grad=True)
    return train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR

def load_all_training_data(case_list = list(range(50))):
    train_water_inject_rate = []
    train_producer_bottom_pressure = []
    train_T = []
    train_OPR = []
    train_WPR = []
    for i in case_list:
        temp_train_water_inject_rate, temp_train_producer_bottom_pressure, temp_train_T, temp_train_OPR, temp_train_WPR = load_training_data(i)
        train_water_inject_rate.append(temp_train_water_inject_rate)
        train_producer_bottom_pressure.append(temp_train_producer_bottom_pressure)
        train_T.append(temp_train_T)
        train_OPR.append(temp_train_OPR)
        train_WPR.append(temp_train_WPR)
    train_water_inject_rate = torch.cat(train_water_inject_rate, dim=0)
    train_producer_bottom_pressure = torch.cat(train_producer_bottom_pressure, dim=0)
    train_T = torch.cat(train_T, dim=0)
    train_OPR = torch.cat(train_OPR, dim=0)
    train_WPR = torch.cat(train_WPR, dim=0)
    return train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR


def train(lambd, weight_decay, lr, model_name):

    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(35)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(35, 45)))

    writer = SummaryWriter(f"./ThreeDTrainedModel/{model_name}/")

    with open('./data/3D_Case/3Dconnection_dict.json', 'r') as f:
        ConnectionsDict = json.load(f)
        ConnectionsDict = {int(key): value for key, value in ConnectionsDict.items()}
    params = {
        'TOTAL_WELL_NUMS': TOTAL_WELL_NUMS,
        'INJECTOR_NUMS': INJECTOR_NUMS,
        'PRODUCER_NUMS': PRODUCER_NUMS,
        'ConnectionsDict': ConnectionsDict,
        'INITIAL_PRESSURE': 2465.6,
        'Bo':1.30500,
        'RockCompressibility': 3.6e-6,
        'mu_w': 0.32,
        'mu_o': 0.366,
        'hid_dim': 128
    }
    model = PINNModel(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    best_test_loss = float('inf')

    for epoch in range(1000000):
        model.train()
        train_f_losses = []
        train_weighted_mse_losses = []
        train_uniform_mse_losses  = []
        train_losses = []

        f_loss, mse_loss_weighted, mse_loss_uniform= model.loss_fn(
            train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
        train_loss = f_loss * lambd + mse_loss_weighted
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_f_losses.append(f_loss.item())
        train_weighted_mse_losses.append(mse_loss_weighted.item())
        train_uniform_mse_losses.append(mse_loss_uniform.item())
        train_losses.append(train_loss.item())
        
        train_f_loss = np.mean(train_f_losses)
        train_weighted_mse_loss = np.mean(train_weighted_mse_losses)
        train_uniform_mse_loss = np.mean(train_uniform_mse_losses)
        train_loss = np.mean(train_losses)

        if epoch % 10 == 0:
            model.eval()
            test_f_losses = []
            test_weighted_mse_losses = []
            test_uniform_mse_losses = []
            test_losses = []

            test_f_loss, test_weighted_mse_loss, test_uniform_mse_loss = model.loss_fn(
                test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
            test_loss = test_f_loss * lambd + test_weighted_mse_loss
        
            test_f_losses.append(test_f_loss.item())
            test_weighted_mse_losses.append(test_weighted_mse_loss.item())
            test_uniform_mse_losses.append(test_uniform_mse_loss.item())
            test_losses.append(test_loss.item())
            
            test_f_loss = np.mean(test_f_losses)
            test_weighted_mse_loss = np.mean(test_weighted_mse_losses)
            test_uniform_mse_loss = np.mean(test_uniform_mse_losses)
            test_loss = np.mean(test_losses)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                if not os.path.exists('./ThreeDTrainedModel'):
                    os.makedirs('./ThreeDTrainedModel')
                torch.save(model.state_dict(), f'./ThreeDTrainedModel/{model_name}.pth')
            # Logging the losses to TensorBoard
            
            writer.add_scalars('Loss/F Loss', {'Train': train_f_loss.item(), "Test": test_f_loss.item()}, epoch)
            writer.add_scalars('Loss/Weighted MSE Loss', {'Train': train_weighted_mse_loss.item(), "Test": test_weighted_mse_loss.item()}, epoch)
            writer.add_scalars('Loss/Uniform MSE Loss', {'Train': train_uniform_mse_loss.item(), "Test": test_uniform_mse_loss.item()}, epoch)
            writer.add_scalars('Loss/Total Loss', {'Train': train_loss.item(), "Test": test_loss.item()}, epoch)
            writer.add_scalars('Learning Rate', {'Learning Rate': optimizer.param_groups[0]['lr']}, epoch)
            print(epoch, 
                  f'train_loss:{train_loss.item():.5f}', 
                  f"train_f_loss:{train_f_loss.item():5f}", 
                  f"train_weighted_mse_loss:{train_weighted_mse_loss.item():5f}", 
                  f"train_uniform_mse_loss:{train_uniform_mse_loss.item():5f}"
                  )
            print(epoch,
                  f"test_loss:{test_loss.item():.5f}", 
                  f"test_f_loss:{test_f_loss.item():5f}", 
                  f"test_weighted_mse_loss:{test_weighted_mse_loss.item():5f}", 
                  f"test_uniform_mse_loss:{test_uniform_mse_loss.item():5f}"
                  )
    
    writer.close()

def trainPureML():
    params = {
        'TOTAL_WELL_NUMS': TOTAL_WELL_NUMS,
        'INJECTOR_NUMS': INJECTOR_NUMS,
        'PRODUCER_NUMS': PRODUCER_NUMS,
        'hid_dim':128
    }
    model = PureMachineLearning(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    
    model_name = f'PureML/3D_PureMachineLearningModel'
    writer = SummaryWriter(f"./logs_pureml/{model_name}/")
    best_test_loss = float('inf')

    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(35)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(35, 45)))
    
    for epoch in range(1000000):
        model.train()
        train_mse_losses = []
        train_losses = []
        
        mse_loss = model.loss_fn(train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
        train_loss =  mse_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_mse_losses.append(mse_loss.item())
        train_losses.append(train_loss.item())
        
        train_mse_loss = np.mean(train_mse_losses)
        train_loss = np.mean(train_losses)

        if epoch % 5 == 0:
            model.eval()
            test_mse_losses = []
            test_losses = []

            test_mse_loss = model.loss_fn(
                test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
            test_loss = test_mse_loss
            
            test_mse_losses.append(test_mse_loss.item())
            test_losses.append(test_loss.item())
            
            test_mse_loss = np.mean(test_mse_losses)
            test_loss = np.mean(test_losses)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                if not os.path.exists(os.path.dirname(f'./ThreeDTrainedModel/{model_name}.pth')):
                    os.makedirs(os.path.dirname(f'./ThreeDTrainedModel/{model_name}.pth'))   
                torch.save(model.state_dict(), f'./ThreeDTrainedModel/{model_name}.pth')
            # Logging the losses to TensorBoard
            
            print(epoch, f'train_loss:{train_loss.item():.5f}', f"train_mse_loss:{train_mse_loss.item():5f}")

            # Log train losses
            writer.add_scalars('Loss/MSE Loss', {'Train': train_mse_loss.item(), 'Test': test_mse_loss.item()}, epoch)
            writer.add_scalars('Learning Rate', {'Learning Rate': optimizer.param_groups[0]['lr']}, epoch)

            print(epoch, f"test_loss:{test_loss.item():.5f}", f"test_mse_loss:{test_mse_loss.item():5f}")
    writer.close()




if __name__ == '__main__':
    trainPureML()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', default=0.0, type=float, required=False)
    parser.add_argument('--weight_decay', default=1e-4,type=float, required=False)
    parser.add_argument('--lr', default=1e-3,type=float, required=False)
    parser.add_argument('--expid', default=1,type=str, required=False)

    args = parser.parse_args()
    lambd = args.lambd
    weight_decay = args.weight_decay
    lr = args.lr
    expid = args.expid

    model_name = f'LINEAR_Weighted_BatchNormal_DROP_expid_{expid}_3D_DownSample_train_35_test_10_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'
    train(lambd, weight_decay, lr, model_name)
