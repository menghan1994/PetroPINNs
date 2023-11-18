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
import pytorch_lightning as pl
import torch.nn.functional as F



class PINNModel(pl.LightningModule):
    def __init__(self, params):
        super(PINNModel, self).__init__()
        TOTAL_WELL_NUMS = params['TOTAL_WELL_NUMS']
        INJECTOR_NUMS = params['INJECTOR_NUMS']
        PRODUCER_NUMS = params['PRODUCER_NUMS']
        self.well_adjecent_index = params['ConnectionsDict']
        self.INITIAL_PRESSURE = params['INITIAL_PRESSURE']
        self.Bo = params['Bo']
        self.RockCompressibility = params['RockCompressibility']
        self.mu_w = params['mu_w']
        self.mu_o = params['mu_o']

        self.hid_dim = params['hid_dim']
        self.number_producer_well = PRODUCER_NUMS
        self.total_wells = TOTAL_WELL_NUMS
        self.input_dim = TOTAL_WELL_NUMS + self.hid_dim

        self.dense1 = nn.Linear(self.input_dim, self.hid_dim)
        self.dense2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.bn1=nn.BatchNorm1d(self.hid_dim)
        self.bn2=nn.BatchNorm1d(self.hid_dim)

        self.dense_output = nn.Linear(
            self.hid_dim, self.total_wells + self.number_producer_well)

        self.WIi = nn.Parameter(torch.tensor(
            [1.0]*len(self.well_adjecent_index), dtype=torch.float32), requires_grad=True)
        self.V0pi = nn.Parameter(torch.tensor(
            [0.2]*len(self.well_adjecent_index), dtype=torch.float32), requires_grad=True)

        self.Swc = nn.Parameter(torch.tensor(
            [-1.0], dtype=torch.float32), requires_grad=True)
        self.Sor = nn.Parameter(torch.tensor(
            [-1.0], dtype=torch.float32), requires_grad=True)

        self.Kro_ = nn.Parameter(torch.tensor(
            [0.00], dtype=torch.float32), requires_grad=True)
        self.Krw_ = nn.Parameter(torch.tensor(
            [0.00], dtype=torch.float32), requires_grad=True)

        self.mu_w = nn.Parameter(torch.tensor([self.mu_w], dtype=torch.float32), requires_grad=False)
        self.mu_oil = nn.Parameter(torch.tensor([self.mu_o], dtype=torch.float32), requires_grad=False)
        self.B_w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)
        self.B_oil = nn.Parameter(torch.tensor([self.Bo], dtype=torch.float32), requires_grad=False)

        self.Cr = nn.Parameter(torch.tensor([self.RockCompressibility], dtype=torch.float32), requires_grad=False)
        self.P0 = nn.Parameter(torch.tensor([self.INITIAL_PRESSURE], dtype=torch.float32), requires_grad=False)

        for producer_ind in self.well_adjecent_index.keys():
            for injector_ind in self.well_adjecent_index[producer_ind]:
                Tij = f'T_{str(producer_ind)}_{injector_ind}'
                if getattr(self, Tij, None) is None:
                    setattr(self, Tij, nn.Parameter(torch.tensor(
                        [0.0], dtype=torch.float32), requires_grad=True))
        self.init_weights()

        self.time_embdding = nn.Linear(1, self.hid_dim)
        self.dropout = nn.Dropout(0.5)
                       
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, inject_well_input, bottom_pressure, inject_input_time):

        x = torch.cat((inject_well_input, bottom_pressure), dim=1)
        time_embdding = self.time_embdding(inject_input_time)
        x = torch.cat((x, time_embdding), dim=1)
        x = torch.tanh(self.bn1(self.dense1(x)))
        x = self.dropout(x)
        x = torch.tanh(self.bn2(self.dense2(x)))
        x = self.dropout(x)
        output = self.dense_output(x)
        return output

    def water_oil_production(self, inject_well_input, bottom_pressure, inject_input_time):

        pressure_water_saturature_pred = self.forward(inject_well_input, bottom_pressure, inject_input_time)

        delta_pressure = torch.exp(pressure_water_saturature_pred[:, :self.total_wells])
    
        water_saturate = torch.sigmoid(pressure_water_saturature_pred[:, self.total_wells:])
    
        S = (water_saturate - torch.sigmoid(self.Swc)) / (1 - torch.sigmoid(self.Swc) - torch.sigmoid(self.Sor))
        kro = (1 - S)**2
        krw = torch.exp(self.Krw_)/torch.exp(self.Kro_) * S**2

        q_w = torch.exp(self.WIi.reshape(1, -1)) * krw / (self.mu_w * self.B_w) * delta_pressure[:, :self.number_producer_well]
        q_o = torch.exp(self.WIi.reshape(1, -1)) * kro / (self.mu_oil * self.B_oil) * delta_pressure[:, :self.number_producer_well]

        return q_w, q_o, kro, krw, delta_pressure, water_saturate


    def loss_fn(self, inject_well_input, bottom_pressure, inject_input_time, OPR, WPR):
        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)
        OPR = OPR.to(self.device)
        WPR = WPR.to(self.device)

        q_w, q_o, kro, krw, delta_pressure, water_saturate = self.water_oil_production(inject_well_input, bottom_pressure, inject_input_time)

        fwpde = []
        fopde = []
        water_first_terms = []
        oil_first_terms = []
        for producer_ind in self.well_adjecent_index.keys():
                        
            injector_indices = self.well_adjecent_index[producer_ind]
            T_values = [torch.exp(getattr(self, f'T_{producer_ind}_{injector}')) for injector in injector_indices]
            Tij_stacked = torch.stack(T_values)

            Twij = krw[:, producer_ind].unsqueeze(-1).unsqueeze(-1) * Tij_stacked / (self.mu_w * self.B_w)
            Toij = kro[:, producer_ind].unsqueeze(-1).unsqueeze(-1) * Tij_stacked / (self.mu_oil * self.B_oil)

            injector_well_ind = torch.tensor(injector_indices)
            pressure_diff = delta_pressure[:, injector_well_ind] - delta_pressure[:, producer_ind].unsqueeze(-1)

            w_frist_term = torch.sum(Twij.squeeze(-1) * pressure_diff, dim=1)
            oil_frist_term = torch.sum(Toij.squeeze(-1) * pressure_diff, dim=1)

            water_first_terms.append(w_frist_term)
            oil_first_terms.append(oil_frist_term)

        w_frist_term = torch.stack(water_first_terms, dim=1).squeeze()
        oil_frist_term = torch.stack(oil_first_terms, dim=1).squeeze()
        
        Vpi = (1 + self.Cr * (self.P0 - (delta_pressure[:, :self.number_producer_well] + bottom_pressure)))*torch.exp(self.V0pi).unsqueeze(0)
        water_third_term = Vpi * water_saturate / self.B_w
        oil_third_term = Vpi * (1 - water_saturate) / self.B_oil

        water_third_t_terms = []
        oil_third_t_terms = []
        for producer_ind in range(water_third_term.size(1)):
            water_third_t_terms.append(torch.autograd.grad(water_third_term[:, producer_ind], inject_input_time, grad_outputs=torch.ones_like(water_third_term[:, producer_ind]), retain_graph=True)[0])
            oil_third_t_terms.append(torch.autograd.grad(oil_third_term[:, producer_ind], inject_input_time, grad_outputs=torch.ones_like(oil_third_term[:, producer_ind]), retain_graph=True)[0])

        water_third_term_t = torch.stack(water_third_t_terms, dim=1).squeeze()
        oil_third_term_t = torch.stack(oil_third_t_terms, dim=1).squeeze()

        fwpde = w_frist_term - q_w - water_third_term_t
        fopde = oil_frist_term - q_o - oil_third_term_t
        weights = self.weights(inject_input_time)

        f_loss = torch.mean(torch.square(fwpde) * weights) + torch.mean(torch.square(fopde) * weights )

        pred_q = torch.cat((q_w, q_o), dim=1)
        real_q = torch.cat((WPR, OPR), dim=1)
        diff = (pred_q - real_q)**2

        mse_loss_weighted = torch.mean(diff * weights)
        mse_loss_uniform = nn.MSELoss()(pred_q, real_q)
        return f_loss, mse_loss_weighted, mse_loss_uniform
    

    def weights(self, inject_input_time):
        max_time = 3623  # Define the maximum inject input time
        min_time = 30    # Define the minimum inject input time

        # Ensure inject_input_time is within the defined range
        inject_input_time = torch.clamp(inject_input_time, min_time, max_time)

        # Linear weights calculation
        # Scale inject_input_time to a range of 0 to 1
        linear_weights = (max_time - inject_input_time) / (max_time - min_time)

        return linear_weights