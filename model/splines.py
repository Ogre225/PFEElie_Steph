#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Spline(nn.Module):
    """ Spline with linear extrapolation """
    def __init__(self, nb_knots=20, x_min=None, x_max=None, is_strictly_increasing=True):
        super().__init__()
        self.nb_knots = nb_knots
        self.x_min = 0 if x_min is None else x_min
        self.x_max = 1 if x_max is None else x_max
        assert self.x_min < self.x_max

        self.alpha = nn.Parameter(torch.zeros(1)) # for inverse transform 
        self.beta = nn.Parameter(torch.zeros(1)) # for inverse transform 

        self.is_strictly_increasing = is_strictly_increasing
        self.eps = 1e-6 # guarantees strict monotonicity

        # Initialization of parameters theta
        if self.is_strictly_increasing:
            self.theta = nn.Parameter(self.y2theta(torch.linspace(self.x_min, self.x_max, self.nb_knots))) 
        else: 
            self.theta = nn.Parameter(torch.linspace(self.x_min, self.x_max, self.nb_knots)) 

    def theta2y(self, theta):
        if not self.is_strictly_increasing:
            return theta
        theta0, theta1 = torch.split(theta, [1, self.nb_knots-1], dim=0)
        return torch.cumsum(torch.cat((theta0, theta1.exp() + self.eps), dim=0), dim=0)

    def y2theta(self, y):
        if not self.is_strictly_increasing:
            return y
        return torch.cat((y[:1], torch.log(y[1:] - y[:-1] - self.eps)), dim=0)

    def forward(self, z, inverse=False):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        device = z.device  # Récupère le dispositif du tenseur d'entrée
     

        if inverse:
            assert self.is_strictly_increasing
            bias = self.alpha.to(device) * z + self.beta.to(device)
            bias=bias.to(device)

        z_input_size = z.size()
        z = z.flatten()
        y = self.theta2y(self.theta)


        y=y.to(device)

        if not inverse:
            z_norm = (self.nb_knots - 1) * (z - self.x_min) / (self.x_max - self.x_min)
            with torch.no_grad():
                i = torch.floor(z_norm).clip(min=0, max=self.nb_knots-2).long()
            y_left = torch.gather(y, dim=0, index=i)
            y_right = torch.gather(y, dim=0, index=i+1)
            t = z_norm - i
            z = y_left * (1-t) + y_right * t
        else:
            with torch.no_grad():
                i = torch.searchsorted(y, z).clip(min=1, max=self.nb_knots-1).long()
            y_left = torch.gather(y, dim=0, index=i-1)
            y_right = torch.gather(y, dim=0, index=i)
            t =  (z - y_left) / (y_right - y_left)
            z = (self.x_max - self.x_min) / (self.nb_knots-1) * (i - 1 + t) + self.x_min
        
        z = z.view(z_input_size)
        return z if not inverse else z + bias