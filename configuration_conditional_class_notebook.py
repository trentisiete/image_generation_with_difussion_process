import torch

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import sys, os
sys.path.insert(0, os.path.abspath("diffusion"))
sys.path.insert(0, os.path.abspath("utils"))
sys.path.insert(0, os.path.abspath("models"))
sys.path.insert(0, os.path.abspath("samplers"))
sys.path.insert(0, os.path.abspath("conditional_samplers"))


from functools import partial

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import functional
import torchvision.transforms as transforms

from torch.optim import Adam

from diffusion.schedules import LinearSchedule, CosineSchedule, NoiseSchedule
from diffusion.sde       import VESDE, VPSDE, SubVPSDE


from diffusion_utilities import (
    plot_image_grid,
    plot_image_grid_color,
    animation_images,
)

# -----------------------------------------------
# (Samplers)
from euler_maruyama_conditional_class import *
from euler_maruyama_conditional_class_vp_sub_vp import *
from predictor_corrector_ve_conditional import *
from predictor_corrector_vp_sub_vp_conditional import *
from probability_flow_ode_conditional import *
from exponential_integrator_conditional import *
# -----------------------------------------------

from models.score_net  import ScoreNet
from models.WideResNet import TimeDependentWideResNet


# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available and will be used.")
else:
    device = torch.device('cpu')
    print("GPU not found, using CPU instead.")


n_threads = torch.get_num_threads()
print('Number of threads: {:d}'.format(n_threads))

# --- Configuraciones ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 32
IMG_CHANNELS = 3

## --Sampler--
N_STEPS = 1000

## --PCSampler--
NUM_CORRECTOR_STEPS = 1
SNR = 0.2
GUIDANCE_FACTOR = 1
EPS_LOSS = 1e-5 # Epsilon para muestrear t en la loss_function y PC

## --Inicialización--
T_END = 1.0 # Tiempo final para la SDE
BETA_MIN = 0.1
BETA_MAX = 20.0
SIGMA_MIN = 0.01 # Para VE SDE
SIGMA_MAX = 50.0  # Para VE SDE
SIGMA = 50.0

S = 0.008 # Para CosineSchedule

def run_configuration(SDE_TYPE, num_samples, target_class, DEVICE):

    if SDE_TYPE == 'VP':
        schedule = LinearSchedule(beta_min=BETA_MIN, beta_max=BETA_MAX)
        sde = VPSDE(schedule=schedule)
    elif SDE_TYPE == 'VE':
        sde = VESDE(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, sigma=SIGMA)
    elif SDE_TYPE == 'SubVP':
        schedule=LinearSchedule(beta_min=BETA_MIN, beta_max=BETA_MAX, T=T_END)
        sde = SubVPSDE(schedule=schedule)
    else:
        raise ValueError(f"SDE type {SDE_TYPE} no soportado.")

    print(f"SDE Type: {SDE_TYPE}")

    if SDE_TYPE == 'VP':
        score_model_path = './checkpoints/scorenet_cifar10_VP_Linear_epoch_50.pth'
    elif SDE_TYPE == 'VE':
        score_model_path = './checkpoints/scorenet_cifar10_VE_epoch_200.pth'
    elif SDE_TYPE == 'SubVP':
        score_model_path = './checkpoints/scorenet_cifar10_SubVP_Linear_epoch_130.pth'
    else:
        raise ValueError(f"SDE type {SDE_TYPE} no soportado.")

    # Re-instanciar la arquitectura
    score_model_loaded = ScoreNet(marginal_prob_std=partial(sde.sigma_t)).to(DEVICE)

    if os.path.exists(score_model_path):
        score_model_loaded.load_state_dict(torch.load(score_model_path, map_location=DEVICE))

        print(f"ScoreNet cargado desde {score_model_path}")
    else:
        print(f"WARN: No se encontró checkpoint de ScoreNet en {score_model_path}")
        


    if SDE_TYPE == 'VP':
        classifier_model_path = './checkpoints_classifier/classifier_cifar10_VP_Linear_epoch_50.pth'
    elif SDE_TYPE == 'VE':
        classifier_model_path = './checkpoints_classifier/classifier_cifar10_VE_No_epoch_100.pth'
    elif SDE_TYPE == 'SubVP':
        classifier_model_path = './checkpoints_classifier/classifier_cifar10_SubVP_Linear_epoch_80.pth'
    else:
        raise ValueError(f"SDE type {SDE_TYPE} no soportado.")

    # Re-instanciar la arquitectura
    classifier_model_loaded = TimeDependentWideResNet(time_emb_dim=128).to(DEVICE) # Asegurar que time_emb_dim coincida

    if os.path.exists(classifier_model_path):
        checkpoint_classifier = torch.load(classifier_model_path, map_location=DEVICE)
        classifier_model_loaded.load_state_dict(checkpoint_classifier['model_state_dict'])
        print(f"Clasificador cargado desde {classifier_model_path}")
    else:
        print(f"WARN: No se encontró checkpoint de Clasificador en {classifier_model_path}")

    if SDE_TYPE == 'VE':
        sampler = ConditionalEulerMaruyamaSampler(
            sde=sde,
            # schedule=schedule,
            score_model=score_model_loaded,
            classifier_model=classifier_model_loaded,
            num_steps=N_STEPS,
            guidance_scale=1,
            t_end_epsilon=1e-5,
            T=T_END,
            stability_eps=1e-5,
            clamp_x_range=None,
            score_clip_value=1000.0,
            grad_clip_value=1.0,
            device=DEVICE
        )
    elif SDE_TYPE == 'VP' or SDE_TYPE == 'SubVP':
        sampler = ConditionalEulerMaruyamaSamplerVP_SubVP(
            sde=sde,
            score_model=score_model_loaded,
            classifier_model=classifier_model_loaded,
            num_steps=N_STEPS,
            guidance_scale=1.5, # Ajusta la fuerza de la guía
            t_end_epsilon=1e-5,
            T=T_END,
            device=DEVICE,
            clamp_x_range=(-1, 1) # Ejemplo para imágenes normalizadas
        )
    else:
        raise ValueError(f"SDE type {SDE_TYPE} no soportado.")

    output_shape = (num_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    print("La salida será", output_shape)
    return sde, score_model_loaded, classifier_model_loaded, sampler, output_shape

# Preguntar al profesor:

    # elif SAMPLER == "PC":
    #     if SDE_TYPE == 'VE':
    #         sampler = ConditionalPCSamplerVE(
    #           sde=sde,
    #           score_model=score_model_loaded,
    #           classifier_model=classifier_model_loaded,
    #           num_steps=N_STEPS,
    #           num_corrector_steps=2,
    #           snr=0.16, # Ajustar según SDE (0.16 es para VE)
    #           guidance_scale=0.8, # Probar diferentes valores (0.5, 1.0, 1.5)
    #           t_end_epsilon=EPS_LOSS,
    #           T=T_END,
    #           device=DEVICE
    #     )
    #     elif SDE_TYPE == 'VP' or SDE_TYPE == 'SubVP':
    #         sampler = ConditionalPCSamplerVPSubVP(
    #           sde=sde,
    #           schedule = schedule,
    #           score_model=score_model_loaded,
    #           classifier_model=classifier_model_loaded,
    #           num_steps=N_STEPS,
    #           num_corrector_steps=NUM_CORRECTOR_STEPS,
    #           snr=SNR,
    #           guidance_scale=GUIDANCE_FACTOR,
    #           T = T_END,
    #           t_eps=EPS_LOSS,
    #           device=DEVICE
    #     )
    # elif SAMPLER == "PFO":
    #     sampler = ConditionalProbabilityFlowODESampler(
    #         sde = sde,
    #         score_model = score_model_loaded,
    #         classifier_model = classifier_model_loaded,
    #         num_steps = 4000,
    #         guidance_scale = 1,
    #         device=DEVICE
    #     )
    # elif SAMPLER == "EI":
    #     sampler = ConditionalExponentialIntegratorSampler(
    #         sde = sde,
    #         score_model = score_model_loaded,
    #         classifier_model = classifier_model_loaded,
    #         schedule = schedule,
    #         num_steps = 8000,
    #         guidance_scale = 1,
    #         device=DEVICE
    #     )
    # print(type(sampler))