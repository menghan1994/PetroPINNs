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
from sklearn.metrics import r2_score
import numpy as np


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
from Train_PINNs_3D import load_all_training_data

def load_PINN_model():
    
    with open('./data/3D_Case/3Dconnection_dict.json', 'r') as f:
        ConnectionsDict = json.load(f)
        ConnectionsDict = {int(key): value for key, value in ConnectionsDict.items()}
    
    TOTAL_WELL_NUMS = 30
    INJECTOR_NUMS = 10
    PRODUCER_NUMS = 20

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
        'hid_dim': 512
    }
    model = PINNModel(params).to(device)
    
    expid = 1
    lr = 1e-3
    lambd = 0.5
    weight_decay = 1e-4
    # model_name = f'Weighted_BatchNormal_DROP_expid_{expid}_3D_DownSample_train_35_test_10_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'
    model_name = f'LINEAR_Weighted_BatchNormal_DROP_expid_{expid}_3D_DownSample_train_35_test_10_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'


    model_save_path = f'./ThreeDTrainedModel/{model_name}.pth'
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    return model

def load_PINN_WPL_model():
    expid = 0
    lr = 0.005
    lambd = 0.0
    weight_decay = 1e-5
    model_name = f'200train_50_test_expid_{expid}_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'
    model_save_path = f'./NewTrainedModel/{model_name}.pth'
    model = PINNModel()
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model


def load_PureML_model():

    TOTAL_WELL_NUMS = 30
    INJECTOR_NUMS = 10
    PRODUCER_NUMS = 20

    params = {
        'TOTAL_WELL_NUMS': TOTAL_WELL_NUMS,
        'INJECTOR_NUMS': INJECTOR_NUMS,
        'PRODUCER_NUMS': PRODUCER_NUMS,
        'hid_dim':128
    }
    model = PureMachineLearning(params).to(device)
    model_save_path = './ThreeDTrainedModel/PureML/3D_PureMachineLearningModel.pth'
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model

def print_out_the_training_loss_and_test_loss():
    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(35)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(35, 45)))
    val_water_inject_rate, val_producer_bottom_pressure, val_T, val_OPR, val_WPR = load_all_training_data(list(range(45, 50)))

    model = load_PINN_model()
    model = model.to(device)
    train_f_loss, _, train_mse_loss = model.loss_fn(
            train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
    test_f_loss, _, test_mse_loss = model.loss_fn(
            test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
    val_f_loss, _, val_mse_loss = model.loss_fn(
            val_water_inject_rate, val_producer_bottom_pressure, val_T, val_OPR, val_WPR)
    
    print('PINN train_f_loss: ', train_f_loss.item())
    print('PINN test_f_loss: ', test_f_loss.item())

    print('PINN train_mse_loss: ', train_mse_loss.item())
    print('PINN test_mse_loss: ', test_mse_loss.item())

    print('PINN val_f_loss: ', val_f_loss.item())
    print('PINN val_mse_loss: ', val_mse_loss.item())


    model = load_PureML_model()
    model = model.to(device)
    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(35)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(35, 45)))
    val_water_inject_rate, val_producer_bottom_pressure, val_T, val_OPR, val_WPR = load_all_training_data(list(range(45, 50)))

    mse_loss = model.loss_fn(train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
    print('PureML train_mse_loss: ', mse_loss.item())
    mse_loss = model.loss_fn(test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
    print('PureML test_mse_loss: ', mse_loss.item())
    mse_loss = model.loss_fn(val_water_inject_rate, val_producer_bottom_pressure, val_T, val_OPR, val_WPR)
    print('PureML val_mse_loss: ', mse_loss.item())



@torch.no_grad()
def visualisation_well_level():
    save_dir = f'./visualisation/3D/'
    model = load_PINN_model()
    for i in tqdm(range(5)):
        case_ind = [i]
        train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(case_ind)
        q_w, q_o, _, _, _, _ = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
        figurename = f'case_{i}'
        visualise_opr_wpr(figurename, save_dir, train_WPR, q_w, train_OPR, q_o)
        # save_path = f'visualisation/temp/'
        real_q_w = train_WPR.detach().cpu().numpy()
        predict_q_w = q_w.detach().cpu().numpy()
        real_q_o = train_OPR.detach().cpu().numpy()
        predict_q_o = q_o.detach().cpu().numpy()
        # np.save(f'{save_path}/real_q_w_{i}.npy', real_q_w)
        # np.save(f'{save_path}/predict_q_w_{i}.npy', predict_q_w)
        # np.save(f'{save_path}/real_q_o_{i}.npy', real_q_o)
        # np.save(f'{save_path}/predict_q_o_{i}.npy', predict_q_o)
        # print('real_q_w: ', real_q_w)
        visualise_opr_wpr(figurename, save_dir, real_q_w, predict_q_w, real_q_o, predict_q_o)


    # for case in range(5):
    #     case_name = f'case{case}'
    #     train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = read_test_data(case_name)
    #     q_w, q_o = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
    #     figurename = f'original_{case}'
    #     visualise_opr_wpr(figurename, save_dir, q_w, train_WPR, q_o, train_OPR)

def visualise_opr_wpr(figurename, save_dir, real_q_w, predict_q_w, real_q_o, predict_q_o):
    # real_q_w = real_q_w.detach().cpu().numpy()
    # predict_q_w = predict_q_w.detach().cpu().numpy()
    # real_q_o = real_q_o.detach().cpu().numpy()
    # predict_q_o = predict_q_o.detach().cpu().numpy()

    colorblind_friendly_colors = [
        "#0173B2", "#DE8F05", "#029E73", "#D55E00",
        "#CC78BC", "#CA9161", "#FBAFE4", "#949494",
        "#ECE133", "#56B4E9"
    ]

    plt.figure(figsize=(10, 8))
    for well_ind in range(real_q_w.shape[-1]):
        color_index = well_ind % len(colorblind_friendly_colors)
        
        plt.plot(predict_q_w[:, well_ind], 
                 label=f'well_{well_ind}_predict', 
                 color=colorblind_friendly_colors[color_index], 
                 marker='*')
        
        plt.plot(real_q_w[:, well_ind], 
                 label=f'well_{well_ind}_real', 
                 color=colorblind_friendly_colors[color_index + 1] 
                        if color_index + 1 < len(colorblind_friendly_colors) else colorblind_friendly_colors[0])
    
    plt.title('Water production rate')
    plt.xlabel('Time (day)')
    plt.ylabel('bbl/day')
    plt.legend()
    if not os.path.exists(f'{save_dir}'):
        os.mkdir(f'{save_dir}')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{figurename}_water_production.pdf')
    plt.close()

    plt.figure(figsize=(10, 8))
    for well_ind in range(real_q_o.shape[-1]):
        color_index = well_ind % len(colorblind_friendly_colors)
        
        plt.plot(predict_q_o[:, well_ind], 
                 label=f'well_{well_ind}_predict', 
                 color=colorblind_friendly_colors[color_index], 
                 marker='*')
        
        plt.plot(real_q_o[:, well_ind], 
                 label=f'well_{well_ind}_real', 
                 color=colorblind_friendly_colors[color_index + 1] 
                        if color_index + 1 < len(colorblind_friendly_colors) else colorblind_friendly_colors[0])

    plt.title('Oil production rate')
    plt.xlabel('Time (day)')
    plt.ylabel('bbl/day')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{figurename}_oil_production.png')
    plt.close()

@torch.no_grad()
def plot_culmultivate_oil_production():
    real_cul = []
    pred_cul = []
    # model = load_PINN_model()
    model = load_PureML_model()
    for i in tqdm(range(50)):
        case_ind = [i]
        train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(case_ind)
        q_w, q_o = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
        real_cul.append(np.sum(train_OPR.detach().cpu().numpy()))
        pred_cul.append(np.sum(q_o.detach().cpu().numpy()))
    
    pred_cul = np.array(pred_cul)
    real_cul = np.array(real_cul)

    np.save(f'./visualisation/temp/3D_pred_cul_pure.npy', pred_cul)
    np.save(f'./visualisation/temp/3D_real_cul_pure.npy', real_cul)



@torch.no_grad()
def error_in_percent_culmultivate_oil_production():
    error_in_percent = []
    model = load_PINN_model()
    for i in tqdm(range(50)):
        case_ind = [i]
        train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(case_ind)
        q_w, q_o = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
        predicted_cumultative = np.sum(q_o.detach().cpu().numpy())
        real_cumultative = np.sum(train_OPR.detach().cpu().numpy())
        print('predicted_cumultative: ', predicted_cumultative, 'real_cumultative: ', real_cumultative)
        error_in_percent.append(np.abs(predicted_cumultative - real_cumultative) / real_cumultative * 100)
    error_in_percent = np.array(error_in_percent)
    print('Mean error in percent train: ', np.mean(error_in_percent[:40]))
    print('Mean error in percent test: ', np.mean(error_in_percent[40:]))
    print('Max error in percent all: ', np.mean(error_in_percent))

def load_PINN_model_with_index(expid):
    lr = 0.005
    lambd = 0.6
    weight_decay = 1e-4
    model_name = f'expid_{expid}_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'
    model_save_path = f'./NewTrainedModel/{model_name}.pth'
    model = PINNModel()
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model

@torch.no_grad()
def visualisation_well_connections():
    from model.PINNModel import well_adjecent_index
    prarms_transmissibility = {}
    for expid in range(1):
        # model = load_PINN_model_with_index(expid)
        model = load_PINN_model()
        model.eval()

        for producer_ind in well_adjecent_index.keys():
            for injector_ind in well_adjecent_index[producer_ind]:
                Tij = f'T_{str(producer_ind)}_{injector_ind}'
                transmissibility = torch.exp(getattr(model, Tij)).item()
                Tij = f'T_{str(producer_ind+1)}_{injector_ind+1}'
                if Tij in prarms_transmissibility:
                    prarms_transmissibility[Tij].append(transmissibility)
                else:
                    prarms_transmissibility[Tij] = [transmissibility]
    # for key in prarms_transmissibility.keys():
        # print(key, 58 * round(np.mean(sorted(prarms_transmissibility[key])[1:-1]), 5))
    
    for key in prarms_transmissibility.keys():
        print(key, prarms_transmissibility[key])
    print(torch.exp(model.WIi))
    print(torch.exp(model.V0pi))

    print('Swc: ', f'{torch.sigmoid(model.Swc).item():.5f}')
    print('Sor: ', f'{torch.sigmoid(model.Sor).item():.5f}')
    print('Krw: ', f'{torch.exp(model.Krw_).item():.5f}')
    print('Kro: ', f'{torch.exp(model.Kro_).item():.5f}')
    print('V0pi: ', [round(item, 5) for item in torch.exp(model.V0pi).numpy().tolist()])
    print('WIi: ', [round(item, 5) for item in torch.exp(model.WIi).numpy().tolist()])
        

def visualisation_traininig_process_loss():
    import json

    with open('./traininglossdata/PureML/test.json', 'r') as f:
        test_loss = np.array(json.load(f))
    with open('./traininglossdata/PureML/train.json', 'r') as f:
        train_loss = np.array(json.load(f))
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss[:, 1], train_loss[:, 2], label='train_loss', c='b')
    plt.plot(test_loss[:, 1], test_loss[:, 2], label='test_loss', c='r')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pure Machine Learning Model')
    plt.savefig('./traininglossdata/PureML.png')
    plt.close()

    lose_type = 'f'
    with open(f'./traininglossdata/PINN/{lose_type}_test.json', 'r') as f:
        test_loss = np.array(json.load(f))
    with open(f'./traininglossdata/PINN/{lose_type}_train.json', 'r') as f:
        train_loss = np.array(json.load(f))

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss[:, 1], train_loss[:, 2], label='train_loss', c='b')
    plt.plot(test_loss[:, 1], test_loss[:, 2], label='test_loss', c='r')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Model Physical Loss')
    plt.savefig(f'./traininglossdata/PINN_{lose_type}.png')
    plt.close()

    lose_type = 'mse'
    with open(f'./traininglossdata/PINN/{lose_type}_test.json', 'r') as f:
        test_loss = np.array(json.load(f))
    with open(f'./traininglossdata/PINN/{lose_type}_train.json', 'r') as f:
        train_loss = np.array(json.load(f))

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss[:, 1], train_loss[:, 2], label='train_loss', c='b')
    plt.plot(test_loss[:, 1], test_loss[:, 2], label='test_loss', c='r')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Model Data Loss')
    plt.savefig(f'./traininglossdata/PINN_{lose_type}.png')
    plt.close()

def calculate_the_cumulative_water_from_inject_into_production_well():
    model = load_PINN_model()
    model = model.to(device)
    all_water_conn = {}
    for case_ind in range(50):
        train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data([case_ind])
        _, water_conn = model.model_analysis(train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)  
        for key in water_conn.keys():
            if key not in all_water_conn.keys():
                all_water_conn[key] = water_conn[key]
            else:
                all_water_conn[key] += water_conn[key]
    sum = 0
    for key in all_water_conn.keys():
        sum += all_water_conn[key]
    
    for key in all_water_conn.keys():
        all_water_conn[key] /= sum
    
    sorted_all_water_conn = sorted(all_water_conn.items(), key=lambda item:item[1], reverse=True)
    print(sorted_all_water_conn)

@torch.no_grad()
def water_saturature_and_pressure():
    colorblind_friendly_colors = [
        "#0173B2", "#DE8F05", "#029E73", "#D55E00",
        "#CC78BC", "#CA9161", "#FBAFE4", "#949494",
        "#ECE133", "#56B4E9"
    ]
    model = load_PINN_model()
    model = model.to(device)
    save_dir = './visualisation/saturarue_pressure/'
    case_ind  = 44
    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data([case_ind])
    pressure, water_saturate, kro, krw = model.formation_pressure_relative_permeability(train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)  
    pressure = pressure.detach().cpu().numpy()
    water_saturate = water_saturate.detach().cpu().numpy()
    krw = krw.detach().cpu().numpy()
    kro = kro.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.plot(pressure[:, i], '.-', label=f'Producer #{i+1}', color=colorblind_friendly_colors[i])
    plt.legend()
    plt.ylabel('Pressure (psi)')
    plt.xlabel('Time (day)')
    plt.title('Formation Pressure')
    plt.savefig(f'{save_dir}/production_{case_ind}_pressure.pdf')
    plt.close()

    plt.figure(figsize=(10, 8))
    for j in range(9, pressure.shape[-1]):
        plt.plot(pressure[:, j], '.-', label=f'Injector #{j-8}', color = colorblind_friendly_colors[j-9])
    plt.legend()
    plt.ylabel('Pressure (psi)')
    plt.xlabel('Time (day)')
    plt.title('Formation Pressure')
    if not os.path.exists(f'{save_dir}'):
        os.makedirs(f'{save_dir}')
    plt.savefig(f'{save_dir}/injection_{case_ind}_pressure.pdf')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.plot(water_saturate[:, i], '.-', label=f'Producer #{i+1}', color=colorblind_friendly_colors[i])
    plt.legend()
    plt.ylabel('Water Saturation')
    plt.xlabel('Time (day)')
    plt.savefig(f'{save_dir}/case_{case_ind}_water_saturate.pdf')

    plt.figure(figsize=(10, 8))
    for i in range(1):
        plt.plot(kro[:, i], label=f'Kro #{i+1}', color = colorblind_friendly_colors[i])
        plt.plot(krw[:, i], label=f'Krw #{i+1}', color=colorblind_friendly_colors[i+1])
    plt.legend()
    
    plt.ylabel('Relative Permeability')
    plt.xlabel('Time (day)')
    plt.savefig(f'{save_dir}/case_{case_ind}_oil_and_water_permeability_produce_1.pdf')

    
    for i in range(9):
        # Plot data on the first y-axis
        fig, ax1 = plt.subplots()
        time = range(pressure.shape[0])
        ax1.plot(time, pressure[:, i], 'g-')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Pressure (psi)', color='g')
        # ax1.tick_params('y', colors='g')

        # Create a second y-axis and plot data on it
        ax2 = ax1.twinx()
        ax2.plot(time, water_saturate[:, i], 'b-')
        ax2.set_ylabel('water saturation', color='b')
        # ax2.tick_params('y', colors='b')

        plt.title("water saturature and pressure")
        plt.savefig(f'{save_dir}/case_{case_ind}_water_saturate_and_pressure_produce_{i+1}.png')
        plt.close()


        plt.plot(pressure[:, i], '.-', label=f'Producer #{i+1}', color=colorblind_friendly_colors[i])

if __name__ == '__main__':
    # error_in_percent_culmultivate_oil_production()
    # visualisation_well_level()
    plot_culmultivate_oil_production()
    # visualisation_well_connections()
    # print_out_the_training_loss_and_test_loss()
    # calculate_the_cumulative_water_from_inject_into_production_well()
    # water_saturature_and_pressure()
    # load_PINN_model()
