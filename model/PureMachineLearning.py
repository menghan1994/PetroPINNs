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

class PureMachineLearning(pl.LightningModule):
    def __init__(self, params):
        super(PureMachineLearning, self).__init__()
        
        TOTAL_WELL_NUMS = params['TOTAL_WELL_NUMS']
        INJECTOR_NUMS = params['INJECTOR_NUMS']
        PRODUCER_NUMS = params['PRODUCER_NUMS']

        self.hid_dim = params['hid_dim']
        self.number_producer_well = PRODUCER_NUMS
        self.total_wells = TOTAL_WELL_NUMS
        self.input_dim = TOTAL_WELL_NUMS + self.hid_dim

        self.dense1 = nn.Linear(self.input_dim, self.hid_dim)
        self.dense2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.bn1=nn.BatchNorm1d(self.hid_dim)
        self.bn2=nn.BatchNorm1d(self.hid_dim)

        self.dense_output = nn.Linear(
            self.hid_dim, 2 * self.number_producer_well)
        
        self.time_embdding = nn.Linear(1, self.hid_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, bottom_pressure, inject_well_input, inject_input_time):

        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)


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
        output = self.forward(bottom_pressure, inject_well_input, inject_input_time)
        q_o = output[:, :self.number_producer_well]
        q_w = output[:, self.number_producer_well:]
        return q_w, q_o

    def loss_fn(self, inject_well_input, bottom_pressure, inject_input_time, real_OPR, real_WPR):
        output = self.forward(bottom_pressure, inject_well_input, inject_input_time)
        # 计算损失
        real_q = torch.cat((real_OPR, real_WPR), dim=1).to(self.device)
        loss = torch.mean((output - real_q)**2)
        return loss