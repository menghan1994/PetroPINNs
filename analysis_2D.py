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
from Train_PINNs_2D import load_all_training_data, read_test_data

def load_PINN_model():
    # model_name = f'best_model_40_train_10_test_lr_0.01_f_loss_{lambd}_weight_decay_{weight_decay}'
    # model_name = f'PINN_Minus_40_train_10_test_lr_0.01_f_loss_{lambd}_weight_decay_{weight_decay}'
    # model_name = f'test_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}_new_experiments'
    # model_name = f'/explore_test_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}_new_experiments'
    expid = 0
    lr = 0.005
    lambd = 0.6
    weight_decay = 1e-5
    # model_name = f'expid_{expid}_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'
    model_name = f'200train_50_test_expid_{expid}_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'
    # model_name = 'PINN_Minus_40_train_10_test_lr_0.01_f_loss_1e-05_weight_decay_1e-05'

    # model_name = f'noise_injection_expid_{expid}_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'

    model_save_path = f'./NewTrainedModel/{model_name}.pth'
    model = PINNModel()
    model.load_state_dict(torch.load(model_save_path))
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
    TOTAL_WELL_NUMS = 13
    INJECTOR_NUMS = 4
    PRODUCER_NUMS = 9

    params = {
        'TOTAL_WELL_NUMS': TOTAL_WELL_NUMS,
        'INJECTOR_NUMS': INJECTOR_NUMS,
        'PRODUCER_NUMS': PRODUCER_NUMS,
        'hid_dim':128
    }
    model = PureMachineLearning(params).to(device)
    model_save_path = './TrainedModel/PureML/2D_PureMachineLearningModel.pth'
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model

def print_out_the_training_loss_and_test_loss():
    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(200)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(200, 250)))

    model = load_PINN_model()
    model = model.to(device)
    train_f_loss, train_mse_loss = model.loss_fn(
            train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
    test_f_loss, test_mse_loss = model.loss_fn(
            test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
    
    print('PINN train_f_loss: ', train_f_loss.item())
    print('PINN test_f_loss: ', test_f_loss.item())

    print('PINN train_mse_loss: ', train_mse_loss.item())
    print('PINN test_mse_loss: ', test_mse_loss.item())

    model = load_PureML_model()
    model = model.to(device)
    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(200)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(200, 250)))
    mse_loss = model.loss_fn(train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
    print('PureML train_mse_loss: ', mse_loss.item())
    mse_loss = model.loss_fn(test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
    print('PureML test_mse_loss: ', mse_loss.item())


@torch.no_grad()
def visualisation_well_level():
    save_dir = f'./visualisation/PLUS/'
    model = load_PINN_model()
    for i in tqdm(range(44, 45)):
        case_ind = [i]
        train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(case_ind)
        q_w, q_o = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
        figurename = f'case_{i}'
        visualise_opr_wpr(figurename, save_dir, train_WPR, q_w, train_OPR, q_o)
        save_path = f'visualisation/temp/'
        real_q_w = train_WPR.detach().cpu().numpy()
        predict_q_w = q_w.detach().cpu().numpy()
        real_q_o = train_OPR.detach().cpu().numpy()
        predict_q_o = q_o.detach().cpu().numpy()
        np.save(f'{save_path}/real_q_w_{i}.npy', real_q_w)
        np.save(f'{save_path}/predict_q_w_{i}.npy', predict_q_w)
        np.save(f'{save_path}/real_q_o_{i}.npy', real_q_o)
        np.save(f'{save_path}/predict_q_o_{i}.npy', predict_q_o)
        print('real_q_w: ', real_q_w)


    # for case in range(5):
    #     case_name = f'case{case}'
    #     train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = read_test_data(case_name)
    #     q_w, q_o = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
    #     figurename = f'original_{case}'
    #     visualise_opr_wpr(figurename, save_dir, q_w, train_WPR, q_o, train_OPR)

def visualise_opr_wpr(figurename, save_dir, real_q_w, predict_q_w, real_q_o, predict_q_o):
    real_q_w = real_q_w.detach().cpu().numpy()
    predict_q_w = predict_q_w.detach().cpu().numpy()
    real_q_o = real_q_o.detach().cpu().numpy()
    predict_q_o = predict_q_o.detach().cpu().numpy()

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
    plt.savefig(f'{save_dir}/{figurename}_oil_production.pdf')
    plt.close()

@torch.no_grad()
def plot_culmultivate_oil_production():
    real_cul = []
    pred_cul = []
    # model = load_PINN_model()
    model = load_PureML_model()
    for i in tqdm(range(250)):
        case_ind = [i]
        train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(case_ind)
        q_w, q_o = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
        real_cul.append(np.sum(train_OPR.detach().cpu().numpy()))
        pred_cul.append(np.sum(q_o.detach().cpu().numpy()))

    pred_cul = np.array(pred_cul)
    real_cul = np.array(real_cul)

    np.save(f'./visualisation/temp/2D_pred_cul_pure.npy', pred_cul)
    np.save(f'./visualisation/temp/2D_real_cul_pure.npy', real_cul)



    # for case in range(5):
    #     case_name = f'case{case}'
    #     train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = read_test_data(case_name)
    #     q_w, q_o = model.water_oil_production(train_water_inject_rate, train_producer_bottom_pressure, train_T)
    #     real_cul.append(np.sum(train_OPR.detach().cpu().numpy()))
    #     pred_cul.append(np.sum(q_o.detach().cpu().numpy()))
    # real_cul = np.array(real_cul)
    # pred_cul = np.array(pred_cul)
    # np.save(f'./visualisation/temp/pred_cul_pinn.npy', pred_cul)
    # np.save(f'./visualisation/temp/real_cul_pinn.npy', real_cul)
    # print('aall')

    # train_sample_num= 200
    # valid_sample_num = 250
    # plt.figure(figsize=(8, 6))
    # plt.scatter(real_cul[:train_sample_num], pred_cul[:train_sample_num], label='Train cases', c = 'b')
    # plt.scatter(real_cul[train_sample_num:valid_sample_num], pred_cul[train_sample_num:valid_sample_num], label='Test cases', c = 'r')
    # plt.scatter(real_cul[valid_sample_num:], pred_cul[valid_sample_num:], label='Validate cases', c = 'k')

    # train_r2 = np.corrcoef(real_cul[:train_sample_num], pred_cul[:train_sample_num])[0, 1]
    # test_r2 = np.corrcoef(real_cul[train_sample_num:valid_sample_num], pred_cul[train_sample_num:valid_sample_num])[0, 1]
    # validation_r2 = np.corrcoef(real_cul[valid_sample_num:], pred_cul[valid_sample_num:])[0, 1]

    # plt.text(0.05, 0.70, f'$r$: {train_r2:.2f} (Train cases) \n   {test_r2:.2f} (Test cases) \n   {validation_r2:.2f} (Validate cases)', 
    #          transform=plt.gca().transAxes, 
    #          fontsize=14)
    # plt.xlabel('Real Cumulative Oil Production (bbl)')
    # plt.ylabel('Predicted Cumulative Oil Production (bbl)')
    # plt.title('Predicted and Real Cumulative Oil Production')
    # plt.legend()
    # # plt.grid()
    # plt.tight_layout()
    # if not os.path.exists('./visualisation/case_level'):
    #     os.mkdir('./visualisation/case_level')
    # plt.savefig('./visualisation/case_level/culmultative_oil_production_Pure.png')
    # plt.close()

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
        

if __name__ == '__main__':
    # error_in_percent_culmultivate_oil_production()
    # visualisation_well_level()
    # visualisation_well_connections()
    # print_out_the_training_loss_and_test_loss()
    # calculate_the_cumulative_water_from_inject_into_production_well()
    # water_saturature_and_pressure()
    # visualisation_well_level
    plot_culmultivate_oil_production()
