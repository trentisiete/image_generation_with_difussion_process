

# Ignorar warnings para mantener la salida limpia
import warnings
warnings.filterwarnings("ignore")

# Bibliotecas estándar
import os
import sys
from functools import partial
from typing import Optional, List, Tuple, Literal # Para type hinting

# Bibliotecas de terceros
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    TensorDataset
)
from torch.optim import Adam
import tqdm
from IPython.display import display, Markdown

paths_to_add = ["diffusion", "utils", "models", "metrics", "samplers", "diffusion_utilities"]
print("Añadiendo rutas locales a sys.path si es necesario:")
for path_name in paths_to_add:
    abs_path = os.path.abspath(path_name)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)
        print(f"  - Añadido: {abs_path}")
    else:
        pass

# SDEs y Schedules
from sde import VESDE, VPSDE, SubVPSDE
from schedules import LinearSchedule, CosineSchedule

# Modelo
from score_net import ScoreNet

# Samplers (Integradores Numéricos)
from euler_maruyama import *
from predictor_corrector import *
from probability_flow_ode import *
from exponential_integrator import *

# Métricas
from fid_for_cifar10 import *
from inception_score import get_inception_score_for_generated_images
from metrics.bpd import calculate_bpd, compute_nll_scipy

# Utilidades (e.g., para visualización)
from diffusion_utilities import (
    plot_image_grid,
    plot_image_evolution,
    plot_image_evolution_pc,
    animation_images,
    plot_image_evolution_color,
    plot_image_grid_color
)


# --- Parámetros de Generación (Sampling) ---
N_IMAGES_TO_GENERATE = 64 # Número de imágenes a generar para visualización/evaluación
SAMPLER_N_STEPS = 500     # Número de pasos para los solvers SDE/ODE (Euler, PC, etc.)
GEN_BATCH_SIZE = 64       # Tamaño del lote para la generación de imágenes

# Selecciona el sampler a usar para generar imágenes (y posiblemente para FID/IS si no se especifica otro)
# Opciones: euler_maruyama_integrator, pc_integrator, ode_sampler, exponential_integrator
SELECTED_SAMPLER = euler_maruyama_integrator
print(f"Sampler seleccionado para generación: {SELECTED_SAMPLER.__name__}")

# --- Parámetros de Evaluación FID (Frechet Inception Distance) ---
CALCULATE_FID = True       # Poner a False para saltar el cálculo de FID
N_FID_SAMPLES = 100      # Número de imágenes generadas para calcular FID (50k es estándar, 10k para pruebas rápidas)
FID_BATCH_SIZE = 128       # Batch size para pasar imágenes por la red Inception (ajustar según VRAM)
FID_DIMS = 2048            # Dimensiones de las activaciones Inception (2048 es estándar)
GEN_BATCH_SIZE_FID = 32
BINARIZE_FID_IMAGES = False # Binarizar imágenes antes de FID (True para MNIST, False para CIFAR-10/color)

# Se puede usar un sampler diferente o con distintos pasos para FID si es necesario
SAMPLER_FOR_FID = euler_maruyama_integrator
SAMPLER_STEPS_FOR_FID = SAMPLER_N_STEPS

# --- Parámetros de Evaluación IS (Inception Score) ---
CALCULATE_IS = True        # Poner a False para saltar el cálculo de IS
IS_SPLITS = 10             # Número de splits para calcular IS (estándar)

# --- Parámetros de Evaluación BPD (Bits Per Dimension) ---
CALCULATE_BPD = True       # Poner a False para saltar el cálculo de BPD/NLL
BPD_BATCH_SIZE = 64        # Batch size para calcular BPD


# --- Instanciar SDE según la configuración ---
def sde_schedule_and_model(MODEL_TYPE, SIGMA_MIN, SIGMA_MAX, SIGMA, BETA_MIN, BETA_MAX, T_END,
                     SCHEDULE_S, DEVICE):
  try:
      if MODEL_TYPE == 'VE':
          sde_instance = VESDE(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, sigma=SIGMA)
          print(f"Inicializado VESDE (sigma_min={SIGMA_MIN}, sigma_max={SIGMA_MAX}, sigma={SIGMA})")
      elif MODEL_TYPE == 'VP_Linear':
          schedule = LinearSchedule(beta_min=BETA_MIN, beta_max=BETA_MAX, T=T_END)
          sde_instance = VPSDE(schedule=schedule)
          print(f"Inicializado VPSDE con Schedule Lineal (beta_min={BETA_MIN}, beta_max={BETA_MAX})")
      elif MODEL_TYPE == 'VP_Cosine':
          schedule = CosineSchedule(T=T_END, s=SCHEDULE_S)
          sde_instance = VPSDE(schedule=schedule)
          print(f"Inicializado VPSDE con Schedule Coseno (s={SCHEDULE_S})")
      elif MODEL_TYPE == 'SubVP_Linear':
          schedule = LinearSchedule(beta_min=BETA_MIN, beta_max=BETA_MAX, T=T_END)
          sde_instance = SubVPSDE(schedule=schedule)
          print(f"Inicializado SubVPSDE con Schedule Lineal (beta_min={BETA_MIN}, beta_max={BETA_MAX})")
      elif MODEL_TYPE == 'SubVP_Cosine':
          schedule = CosineSchedule(T=T_END, s=SCHEDULE_S)
          sde_instance = SubVPSDE(schedule=schedule)
          print(f"Inicializado SubVPSDE con Schedule Coseno (s={SCHEDULE_S})")
      else:
          raise ValueError(f"Tipo de modelo desconocido: {MODEL_TYPE}")

  except Exception as e:
      print(f"Error durante la inicialización del modelo o SDE: {e}")
      raise
      return None,None, None
  
  # Cargamos el modelo desde un checkpoint si existe
  checkpoints_files = {'VE': 200, 'VP_Linear': 50, 'VP_Cosine': 100, 'SubVP_Linear': 130, 'SubVP_Cosine': 20}

  score_model_path = f'./checkpoints/scorenet_cifar10_{MODEL_TYPE}_epoch_{checkpoints_files[MODEL_TYPE]}.pth'
  print(score_model_path)

  # Load state_dict with strict=False
  score_model = ScoreNet(marginal_prob_std=partial(sde_instance.sigma_t)).to(DEVICE)

  if os.path.exists(score_model_path):
      score_model.load_state_dict(torch.load(score_model_path, map_location=DEVICE))

      print(f"ScoreNet cargado desde {score_model_path}")
  else:
      print(f"WARN: No se encontró checkpoint de ScoreNet en {score_model_path}")


  # score_model.load_state_dict(torch.load(f'./checkpoints/scorenet_cifar10_{MODEL_TYPE}_epoch_{checkpoints_files[MODEL_TYPE]}.pth', map_location=DEVICE))
  score_model.eval()
  print("Checkpoint loaded successfully.")
  if MODEL_TYPE == 'VE':
    return sde_instance, score_model
  else:
    return sde_instance,schedule, score_model
 



def generate_from_sampler(sde_instance, score_model, SAMPLERS, N_IMAGES, IMG_CHANNELS, IMG_SIZE, MODEL_TYPE,DEVICE,T=1,
                          PC_SAMPLER_STEPS=1000, PC_SNR=0.27, PC_CORRECTOR_STEPS=2, PC_T_FINAL=1e-4,
                          EXP_EULER_ODE_T_FINAL=1e-4, EXP_EULER_ODE_STEPS=2000,
                          P_FLOW_ODE_T_FINAL=1e-3, P_FLOW_ODE_STEPS=2000,
                          EULER_MARUYAMA_STEPS=1000, EULER_MARUYAMA_T_FINAL=1e-4,
                          N_INTERMEDIATE_STEPS_PLOT_EVO=10, ):
  
  image_shape = (N_IMAGES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
  final_images = dict()

  if 1 in SAMPLERS:
    display(Markdown(f"#Generando Imagenes con Predictor Corrector"))
    
    pc_sampler = PredictorCorrectorSampler(
        sde=sde_instance,
        score_model=score_model,
        num_steps=PC_SAMPLER_STEPS,             # N
        num_corrector_steps= PC_CORRECTOR_STEPS,      # M
        snr=PC_SNR,                   # r
        t_end=PC_T_FINAL,                 # Tiempo final
        device=DEVICE,
    )
    # Generar muestras
    with torch.no_grad():
        synthetic_images = pc_sampler.sample(shape=image_shape)

    # Pasos intermedios plot
    num_puntos_totales = N_INTERMEDIATE_STEPS_PLOT_EVO + 2
    pasos_intermedios = np.linspace(start=0, stop=PC_SAMPLER_STEPS, num=num_puntos_totales, dtype=int)
    
    display(Markdown(f"##Evolución de imagenes con Predictor-Corrector"))
    
    _ = plot_image_evolution_pc(
    images=synthetic_images[1].cpu(),
    n_images=N_IMAGES,
    n_intermediate_steps=pasos_intermedios,
    figsize=(12, 2 * N_IMAGES) # Ajustar tamaño
    )
    plt.show()
    final_images['Predictor_corrector'] = synthetic_images[1][...,-1]
    
  if 2 in SAMPLERS:
    display(Markdown(f'#Generando Imagenes con Exponential Integrator'))
    # 1) Inicializar x(T) desde el prior
    sigma_T = sde_instance.sigma_t(torch.tensor([T], device=DEVICE))
    #    Generar ruido N(0,1)
    noise = torch.randn(N_IMAGES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)
    #    Escalar para obtener N(0, sigma_T^2)
    image_T = noise * sigma_T

    # 2) Ejecutar el integrador Exponential Euler ODE
    with torch.no_grad():
        try:
            times, synthetic_images_t_exp_euler = exponential_euler_ode_sampler(
                x_start = image_T,
                sde = sde_instance,
                score_model = score_model,
                t_start = T,
                t_end = EXP_EULER_ODE_T_FINAL,
                n_steps = EXP_EULER_ODE_STEPS,
                device = DEVICE
            )

            # La trayectoria completa está en synthetic_images_t_exp_euler
            # La imagen final generada está en el último paso de tiempo
            final_image_exp_euler = synthetic_images_t_exp_euler[..., -1]

        except ValueError as e:
            print(f"Error al ejecutar el sampler Exponential Euler: {e}")
            print("Asegúrate de estar usando una SDE de tipo VP o SubVP.")
    
    # Pasos intermedios plot
    num_puntos_totales = N_INTERMEDIATE_STEPS_PLOT_EVO + 2
    pasos_intermedios = np.linspace(start=0, stop=EXP_EULER_ODE_STEPS, num=num_puntos_totales, dtype=int)

    display(Markdown(f"##Evolución de las imagenes Generadas con Exponential-Integrator ODE Euler"))
    
    _ = plot_image_evolution_pc(
    images=synthetic_images_t_exp_euler.cpu(),
    n_images=N_IMAGES,
    n_intermediate_steps=pasos_intermedios,
    figsize=(12, 2 * N_IMAGES) # Ajustar tamaño
    )
    plt.show()
    final_images['Exponential Integrator'] = final_image_exp_euler
  if 3 in SAMPLERS:
    display(Markdown('#Generando Imagenes con Probability Flow ODE'))
    # 1) Inicializar x(T) desde el prior. Esto no cambia.
    sigma_T = sde_instance.sigma_t(torch.tensor([T], device=DEVICE))
    #    Generar ruido N(0,1)
    noise = torch.randn(N_IMAGES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)
    #    Escalar para obtener N(0, sigma_T^2)
    image_T = noise * sigma_T

    # 2) Ejecutar el integrador ODE
    with torch.no_grad():
        times, synthetic_images_t_ode = probability_flow_ode_integrator(
            x_start = image_T,
            sde = sde_instance,
            score_model = score_model,
            t_start = T,
            t_end = P_FLOW_ODE_T_FINAL, # Tiempo final
            n_steps = P_FLOW_ODE_STEPS,
            device = DEVICE
        )

    # synthetic_images_t_ode contendrá la trayectoria completa.
    # La imagen final generada estará en synthetic_images_t_ode[..., -1]
    final_image_ode = synthetic_images_t_ode[..., -1]

    # Pasos intermedios plot
    num_puntos_totales = N_INTERMEDIATE_STEPS_PLOT_EVO + 2
    pasos_intermedios = np.linspace(start=0, stop=P_FLOW_ODE_STEPS, num=num_puntos_totales, dtype=int)

    display(Markdown(f"##Evolución de las imagenes Generadas con Probability Flow ODE"))
    _ = plot_image_evolution_pc(
    images=synthetic_images_t_ode.cpu(),
    n_images=N_IMAGES,
    n_intermediate_steps=pasos_intermedios,
    figsize=(12, 2 * N_IMAGES) # Ajustar tamaño
    )
    plt.show()
    final_images["Probability Flow ODE"] = final_image_ode
  if 4 in SAMPLERS:
    display(Markdown('#Generando Imagenes con Euler Maruyama'))
    # 1) Cálculo de sigma^2(T) con la fórmula dada
    sigma_cuadrado_evaluado_en_T = sde_instance.sigma_t(T) ** 2

    # 2) Creación de un tensor de ruido con media 0 y varianza 1
    noise = torch.randn(N_IMAGES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)

    # 3) Escalar el ruido para que tenga varianza sigma_cuadrado_evaluado_en_T
    image_T = noise * np.sqrt(sigma_cuadrado_evaluado_en_T)
    with torch.no_grad():
        times, synthetic_images_t_euler_maruyama = euler_maruyama_integrator(
            image_T,
            t_0 = T,
            t_end = EULER_MARUYAMA_T_FINAL,
            n_steps = EULER_MARUYAMA_STEPS,
            drift_coefficient = partial(
                sde_instance.backward_drift_coefficient,
                score_model = score_model,
            ),
            diffusion_coefficient = sde_instance.diffusion_coefficient,
        )

    # Pasos intermedios plot
    num_puntos_totales = N_INTERMEDIATE_STEPS_PLOT_EVO + 2
    pasos_intermedios = np.linspace(start=0, stop=EULER_MARUYAMA_STEPS, num=num_puntos_totales, dtype=int)

    display(Markdown(f"##Evolución de las imagenes Generadas con Euler Maruyama"))

    _ = plot_image_evolution_pc(
    images=synthetic_images_t_euler_maruyama.cpu(),
    n_images=N_IMAGES,
    n_intermediate_steps=pasos_intermedios,
    figsize=(12, 2 * N_IMAGES) # Ajustar tamaño
    )
    plt.show()
    final_images['Euler Maruyama'] = synthetic_images_t_euler_maruyama[...,-1]
    return final_images


def plot_tensor_images(
    tensor_batch: torch.Tensor,
    figsize_per_image: Tuple[float, float] = (3, 3),
    normalize_method: Literal['clip', 'minmax', 'none'] = 'clip',
    titles: Optional[List[str]] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Muestra las imágenes contenidas en un tensor de PyTorch [B, C, H, W].

    Args:
        tensor_batch: El tensor de PyTorch con forma [Batch, Channels, Height, Width].
                      Se espera que Channels (C) sea 1 (escala de grises) o 3 (RGB).
        figsize_per_image: Tamaño (ancho, alto) en pulgadas para cada subplot de imagen.
        normalize_method: Método para ajustar el rango de valores para la visualización:
                            'clip': Corta los valores al rango [0, 1]. (Predeterminado)
                            'minmax': Escala cada imagen individualmente a [0, 1].
                            'none': No aplica ninguna normalización explícita (confía en imshow).
        titles: Una lista opcional de títulos, uno para cada imagen del lote.

    Returns:
        Una tupla (fig, axs) con la figura y los ejes de Matplotlib.

    Raises:
        TypeError: Si la entrada no es un tensor de PyTorch.
        ValueError: Si el tensor no tiene 4 dimensiones, si el número de canales
                    no es 1 o 3, o si la lista de títulos no coincide con el tamaño del lote.
    """
    # --- Validaciones de Entrada ---
    if not isinstance(tensor_batch, torch.Tensor):
        raise TypeError("La entrada debe ser un tensor de PyTorch.")
    if tensor_batch.dim() != 4:
        raise ValueError(f"El tensor debe tener 4 dimensiones [B, C, H, W], pero tiene {tensor_batch.dim()}")

    B, C, H, W = tensor_batch.shape

    if B == 0:
        print("Warning: El tensor no contiene imágenes (Batch size es 0).")
        # Devolver una figura vacía o manejar como prefieras
        fig, ax = plt.subplots()
        return fig, np.array([[ax]]) # Devolver array consistente con el caso B > 0

    if C not in [1, 3]:
        raise ValueError(f"El número de canales (C={C}) debe ser 1 (escala de grises) o 3 (RGB).")

    if titles is not None and len(titles) != B:
        raise ValueError(f"La lista de títulos ({len(titles)}) debe tener la misma longitud que el tamaño del lote ({B}).")

    # --- Crear Figura ---
    # Crea una fila con B columnas para las imágenes
    fig, axs = plt.subplots(1, B, figsize=(B * figsize_per_image[0], figsize_per_image[1]), squeeze=False)
    # squeeze=False asegura que axs siempre sea un array 2D (1xB), incluso si B=1

    # --- Procesar y Mostrar cada Imagen ---
    for i in range(B):
        ax = axs[0, i] # Acceder al eje correcto en la primera (y única) fila
        img_chw = tensor_batch[i] # Seleccionar imagen i -> [C, H, W]

        # Permutar dimensiones y manejar escala de grises vs RGB
        if C == 1: # Escala de grises
            # Quitar la dimensión del canal (squeeze) -> [H, W]
            img_permuted = img_chw.squeeze(0)
            cmap = 'gray' # Usar mapa de colores gris
        else: # RGB
            # Permutar a -> [H, W, C]
            img_permuted = img_chw.permute(1, 2, 0)
            cmap = None # imshow detecta RGB automáticamente, no necesita cmap

        # Convertir a NumPy (asegurándose de que está en CPU y sin gradientes)
        img_np = img_permuted.detach().cpu().numpy()

        # Normalizar para visualización según el método elegido
        if normalize_method == 'clip':
            # Cortar valores fuera del rango [0, 1]
            img_display = np.clip(img_np, 0, 1)
        elif normalize_method == 'minmax':
            # Escalar los valores de esta imagen específica a [0, 1]
            min_val = img_np.min()
            max_val = img_np.max()
            if max_val > min_val:
                img_display = (img_np - min_val) / (max_val - min_val)
            else:
                # Si la imagen es constante, mostrarla como negra (o gris medio)
                img_display = np.zeros_like(img_np)
        elif normalize_method == 'none':
             # No hacer nada, confiar en la normalización automática de imshow
            img_display = img_np
        else:
             raise ValueError(f"Método de normalización desconocido: {normalize_method}")

        # Mostrar imagen
        ax.imshow(img_display, cmap=cmap)

        # Poner título
        if titles:
            ax.set_title(titles[i])
        else:
            # Título por defecto si no se proporcionan títulos
            ax.set_title(f"Image {i}")

        ax.axis('off') # Ocultar los ejes

    plt.tight_layout() # Ajustar espaciado
    return fig, axs # Devolver figura y ejes para posible uso posterio
    
def final_images_plot(final_images):
  from IPython.display import display, Markdown
  for i in final_images.keys():
    display(Markdown(f"#Imagenes finales Generadas por {i}"))
    # plot_tensor_images(final_images[i])
    plot_image_grid_color(final_images[i], figsize=(10,8), n_cols=len(final_images[i]), n_rows=1)
    plt.show()
      

def bdp_cifar(sde_instance, score_model, NUM_WORKERS, IMG_CHANNELS, IMG_SIZE, DEVICE, partition_subset= 6):
  full_cifar10_test_dataset = datasets.CIFAR10(
      root='./data',       # Directorio donde guardar/buscar los datos
      train=False,       # Usar el conjunto de test para evaluación
      download=True,     # Descargar si no existe
      transform=ToTensor() # Escala imágenes a [0, 1]
  )
  print(f"Full CIFAR-10 test set size: {len(full_cifar10_test_dataset)}")

  # --- Crear Subset para usar solo la mitad del dataset ---
  total_size = len(full_cifar10_test_dataset)
  subset_size = total_size // partition_subset # División entera para obtener aproximadamente la mitad


  # Elegir índices aleatorios sin reemplazo
  indices = np.random.choice(total_size, subset_size, replace=False)

  # Crear el objeto Subset
  subset_dataset = Subset(full_cifar10_test_dataset, indices)
  print(f"Created subset with {len(subset_dataset)} images (first half of test set).")
  # --- FIN NUEVA SECCIÓN ---


  # --- Crear DataLoader ---
  eval_dataset = subset_dataset

  eval_loader = DataLoader(
      eval_dataset,
      batch_size=BPD_BATCH_SIZE,
      shuffle=False,      # No barajar para evaluación determinista
      num_workers=NUM_WORKERS,
      pin_memory=True if DEVICE == 'cuda' else False # Acelera transferencia a GPU
  )

  # --- Dimensiones CIFAR-10 ---
  # --- Parámetros del Dataset ---
  dimensions = IMG_CHANNELS * IMG_SIZE * IMG_SIZE
  print(f"Image dimensions: C={IMG_CHANNELS}, H={IMG_SIZE}, W={IMG_SIZE} => Total={dimensions}")

  # --- Calcular NLL promedio usando Scipy ---
  print("\nIniciando cálculo de NLL con compute_nll_scipy...")
  # Parámetros para el solver ODE (pueden necesitar ajuste)
  t_start_nll = 1e-5 # Tiempo inicial pequeño (cercano a 0)
  t_end_nll = 1 # Tiempo final (usualmente 1.0), tomar de la SDE
  rtol_nll = 1e-4    # Tolerancia relativa
  atol_nll = 1e-4    # Tolerancia absoluta
  method_nll = 'RK45' # Método del solver ODE

  avg_nll = compute_nll_scipy(
      data_loader=eval_loader,
      sde=sde_instance,
      score_model=score_model,
      device=DEVICE,
      t_start=t_start_nll,
      t_end=t_end_nll,
      rtol=rtol_nll,
      atol=atol_nll,
      method=method_nll,
      do_dequantize=True # ¡Importante activar la dequantización para datasets discretos!
  )

  # --- Calcular y mostrar BPD ---
  if avg_nll is not None:
      print(f"\nCálculo NLL finalizado.")
      print(f"Average NLL (base e): {avg_nll:.4f}")

      # Asegurar que avg_nll es un tensor para calculate_bpd si la función lo requiere
      if not isinstance(avg_nll, torch.Tensor):
          avg_nll_tensor = torch.tensor(avg_nll, device='cpu')
      else:
          avg_nll_tensor = avg_nll.cpu()

      # Calcular BPD
      bpd_value = calculate_bpd(avg_nll_tensor, dimensions)
      
      display(Markdown(f"### Bits Per Dimension (BPD): {bpd_value.item():.4f}"))
      return bpd_value.item()

  else:
      print("\nError: No se pudo calcular el NLL promedio.")
      

def fid_cifar10(sde_instance, score_model, SAMPLER_FOR_FID, N_FID_SAMPLES, GEN_BATCH_SIZE_FID, IMG_CHANNELS, IMG_SIZE,T, t_final, MODEL_TYPE, SAMPLER_N_STEPS, DEVICE):
  # --- 1. Generar Imágenes Sintéticas ---
  print(f"Generando {N_FID_SAMPLES} imágenes sintéticas con {MODEL_TYPE} para FID...")
  # Nota: Asume que generate_samples_for_fid maneja el bucle de batches internamente
  generated_images_fid = generate_samples_for_fid(
    score_model_sde=score_model,       # Modelo cargado/entrenado
    sde=sde_instance,                  # SDE instanciada
    sampler=SAMPLER_FOR_FID,           # Sampler seleccionado
    n_samples=N_FID_SAMPLES,
    batch_size=GEN_BATCH_SIZE_FID,
    img_channels= IMG_CHANNELS,
    img_size = IMG_SIZE,
    t0= T,# Tiempo inicial SDE (forward, ej: 1.0)
    t1= t_final,                           # Tiempo final SDE (sampling, cerca de 0)
    n_steps=SAMPLER_N_STEPS,           # Pasos del solver
    device=DEVICE,
    model_type=MODEL_TYPE,         # Tipo de modelo
  )
  print(f"Imágenes generadas. Tensor shape: {generated_images_fid.shape}")
  print(f"Preparando {N_FID_SAMPLES} imágenes reales aleatorias de CIFAR-10 para FID...")

  # --- 1. Cargar el dataset CIFAR-10 (Test set) ---
  # Es estándar usar el conjunto de test para evaluación FID.
  # Usar ToTensor() para escalar a [0, 1], asumiendo que las imágenes generadas
  # también estarán (o serán llevadas) a este rango antes del cálculo FID.
  try:
      cifar10_test_dataset_for_fid = datasets.CIFAR10(
          root='./data',       # Directorio donde guardar/buscar los datos CIFAR-10
          train=False,       # Usar conjunto de test
          download=True,     # Descargar si no existe
          transform=ToTensor() # Escala imágenes a [0, 1] Tensor[C, H, W]
      )
      print(f"Dataset CIFAR-10 test cargado. Tamaño total: {len(cifar10_test_dataset_for_fid)}")
  except Exception as e:
      raise RuntimeError(f"Error al cargar CIFAR-10: {e}. Asegúrate de que torchvision esté instalado y tengas conexión a internet si necesita descarga.")

  # --- 2. Determinar número de muestras y seleccionar índices aleatorios ---
  num_total_real = len(cifar10_test_dataset_for_fid)

  # Ajustar si se piden más muestras de las disponibles
  if N_FID_SAMPLES > num_total_real:
      warnings.warn(
          f"Se solicitaron N_FID_SAMPLES={N_FID_SAMPLES}, pero el dataset CIFAR-10 test solo tiene {num_total_real}. "
          f"Se usarán {num_total_real} imágenes reales para FID."
      )
      num_real_to_take = num_total_real
  else:
      num_real_to_take = N_FID_SAMPLES

  # Seleccionar 'num_real_to_take' índices aleatorios sin reemplazo del dataset
  # torch.randperm es una forma eficiente de hacerlo
  indices = torch.randperm(num_total_real)[:num_real_to_take]
  print(f"Seleccionando {num_real_to_take} imágenes reales al azar de CIFAR-10 test.")

  # --- 3. Cargar las imágenes correspondientes a los índices seleccionados ---
  real_images_list_fid = []
  print(f" Cargando {num_real_to_take} imágenes reales...")

  # Iterar sobre los índices aleatorios y obtener las imágenes
  # (Se ignora la etiqueta '_')
  for idx in tqdm.tqdm(indices, desc="Cargando imágenes reales"): # Añadir tqdm para barra de progreso
      real_img, _ = cifar10_test_dataset_for_fid[idx.item()] # .item() para convertir índice tensor a int
      real_images_list_fid.append(real_img)

  # Verificar si se cargaron imágenes
  if not real_images_list_fid:
      # Esto solo ocurriría si num_real_to_take fuera 0
      raise ValueError("No se pudieron cargar imágenes reales de CIFAR-10 (la lista está vacía).")

  # --- 4. Apilar la lista de tensores en un solo tensor y mover al dispositivo ---
  real_images_tensor_fid = torch.stack(real_images_list_fid).to(DEVICE)
  print(f"Imágenes reales preparadas. Tensor shape: {real_images_tensor_fid.shape}, Device: {real_images_tensor_fid.device}")
  print("Creando DataLoaders para FID...")

  # --- 1. Verificar y asegurar consistencia en el número de muestras ---
  num_gen = generated_images_fid.shape[0]
  num_real = real_images_tensor_fid.shape[0] # Usar shape[0] del tensor real ya cargado

  # Comprobar si los números son diferentes (num_real ya debería ser == num_real_to_take)
  if num_gen != num_real:
      min_count = min(num_gen, num_real)
      warnings.warn(
          f"El número de imágenes generadas ({num_gen}) y reales ({num_real}) difiere. "
          f"FID se calculará usando el mínimo ({min_count}) de cada conjunto para asegurar una comparación justa."
      )
      # --- Truncar ambos tensores al tamaño mínimo (ACTIVADO) ---
      generated_images_fid = generated_images_fid[:min_count]
      real_images_tensor_fid = real_images_tensor_fid[:min_count]
      print(f" Ambos conjuntos de imágenes truncados a {min_count} muestras.")
  elif num_gen == 0:
      raise ValueError("No hay imágenes generadas para calcular FID.")


  # --- 2. Mover tensores al dispositivo correcto (DEVICE) si no lo están ---
  # Es importante que ambos estén en el mismo dispositivo para TensorDataset/DataLoader
  if generated_images_fid.device != DEVICE:
      generated_images_fid = generated_images_fid.to(DEVICE)
      print(f" Tensor de imágenes generadas movido a {DEVICE}.")
  if real_images_tensor_fid.device != DEVICE:
      # El paso anterior ya debería haberlo puesto en DEVICE, pero comprobamos por si acaso
      real_images_tensor_fid = real_images_tensor_fid.to(DEVICE)
      print(f" Tensor de imágenes reales movido a {DEVICE}.")


  # --- 3. Crear TensorDatasets ---
  # TensorDataset simplemente envuelve un tensor, facilitando su uso con DataLoader.
  # Espera que la primera dimensión sea el número de muestras.
  try:
      real_fid_dataset = TensorDataset(real_images_tensor_fid)
      gen_fid_dataset  = TensorDataset(generated_images_fid)
      print(f"TensorDatasets creados. Tamaño real: {len(real_fid_dataset)}, Tamaño generado: {len(gen_fid_dataset)}")
  except Exception as e:
      print(f"Error creando TensorDataset. Verifica las formas de los tensores:")
      print(f" Real shape: {real_images_tensor_fid.shape}")
      print(f" Generadas shape: {generated_images_fid.shape}")
      raise e

  # --- 4. Crear DataLoaders ---
  # shuffle=False es crucial para evaluación determinista y comparación.
  # num_workers=0 y pin_memory=False suelen ser adecuados para TensorDataset
  # ya que los datos ya están en memoria (posiblemente GPU).
  try:
      real_fid_loader = DataLoader(
          real_fid_dataset,
          batch_size=FID_BATCH_SIZE, # Usa la variable definida previamente
          shuffle=False,
          num_workers=0,
          pin_memory=False
      )
      gen_fid_loader = DataLoader(
          gen_fid_dataset,
          batch_size=FID_BATCH_SIZE, # Usa la variable definida previamente
          shuffle=False,
          num_workers=0,
          pin_memory=False
      )
      print(f"DataLoaders creados con batch_size={FID_BATCH_SIZE}.")
  except NameError:
      raise NameError("La variable 'FID_BATCH_SIZE' no está definida. Asegúrate de definirla antes de este bloque.")
  except Exception as e:
      print(f"Error creando DataLoaders: {e}")
      raise e


  # Ahora 'real_fid_loader' y 'gen_fid_loader' están listos para ser pasados
  # a la función calculate_fid(real_fid_loader, gen_fid_loader, device=DEVICE, ...)

  # --- 5. Calcular FID ---

  # Obtener el número de muestras (deberían ser iguales tras el paso anterior)
  # Es buena idea comprobar que no sea cero.
  try:
      num_samples_for_fid_calc = len(real_fid_loader.dataset)
      if num_samples_for_fid_calc == 0:
          raise ValueError("El DataLoader real para FID está vacío. No se puede calcular FID.")
      # Podrías añadir una comprobación extra si quieres:
      # assert len(real_fid_loader.dataset) == len(gen_fid_loader.dataset), "Los DataLoaders FID no tienen el mismo tamaño"
  except Exception as e:
      raise ValueError(f"Error al obtener el tamaño del dataset del DataLoader: {e}")


  print(f"\nCalculando FID usando {num_samples_for_fid_calc} imágenes reales y {num_samples_for_fid_calc} generadas...")

  # Llamada a la función externa que calcula FID (definida previamente)
  # Asegúrate de que DEVICE y FID_DIMS están definidas.
  try:
      fid_score = calculate_fid(
          real_fid_loader,          # DataLoader de imágenes reales CIFAR-10
          gen_fid_loader,           # DataLoader de imágenes generadas
          device=DEVICE,            # Dispositivo (cuda/cpu)
          dims=FID_DIMS,            # Dimensiones de Inception (ej. 2048)
          num_samples=num_samples_for_fid_calc # Número de muestras a usar por calculate_activation_statistics
          # El argumento batch_size no es necesario aquí, ya está implícito en los DataLoaders
      )
  except NameError as e:
      # Captura error si DEVICE o FID_DIMS no están definidas
      raise NameError(f"Error: Una variable necesaria (DEVICE o FID_DIMS) no está definida. Detalle: {e}")
  except Exception as e:
      # Captura otros posibles errores durante el cálculo
      print(f"Ocurrió un error durante calculate_fid: {e}")
      raise e

  # Imprimir el resultado final (adaptado para CIFAR-10, sin TARGET_DIGIT)
  display(Markdown(f"### FID Score ({MODEL_TYPE}): {fid_score:.4f}"))
  return fid_score


  # --- 6. Visualizar Muestras Generadas (Aplicando Escalado [-1,1] -> [0,1]) ---

  # Verificar que la variable con imágenes generadas existe y no está vacía
  if 'generated_images_fid' in locals() and isinstance(generated_images_fid, torch.Tensor) and generated_images_fid.numel() > 0:
      print("\n--- Visualizando Algunas Imágenes Generadas (Escaladas a [0, 1] para mostrar) ---")
      n_rows_vis = 4  # Número de filas en la cuadrícula
      n_cols_vis = 8  # Número de columnas en la cuadrícula
      # Calcular cuántas imágenes mostrar, como máximo n_rows * n_cols
      num_to_plot = min(len(generated_images_fid), n_rows_vis * n_cols_vis)

      if num_to_plot > 0:
          # 1. Seleccionar el subconjunto de imágenes a plotear y mover a CPU
          images_subset_raw = generated_images_fid[:num_to_plot].cpu()

          # --- INICIO DE LA MODIFICACIÓN ---
          # 2. Escalar las imágenes del rango asumido [-1, 1] al rango [0, 1]
          #    Esta es la transformación estándar para visualizar datos normalizados en [-1, 1].
          #    Si tus datos originales NO están en [-1, 1], necesitarás ajustar esto
          #    o usar un método diferente (ej. min-max por lote, o clip).
          images_subset_scaled = (images_subset_raw + 1.0) / 2.0

          # 3. (Recomendado) Aplicar clamp (cortar) para asegurar que estrictamente estén en [0, 1]
          #    Esto maneja pequeños errores numéricos que podrían dejar valores como 1.00001 o -0.00001
          images_for_plot = torch.clamp(images_subset_scaled, 0.0, 1.0)
          # --- FIN DE LA MODIFICACIÓN ---

          # 4. Llamar a la función de visualización con las imágenes YA ESCALADAS
          try:
              # Asumiendo que tienes una función plot_image_grid como en tu código original
              # Reemplaza 'plot_image_grid' por 'plot_tensor_images' si es la que definimos antes
              plot_image_grid( # O plot_tensor_images
                  images=images_for_plot, # <--- Pasar el tensor escalado y cortado
                  n_rows=n_rows_vis,
                  n_cols=n_cols_vis,
                  figsize=(n_cols_vis * 1.2, n_rows_vis * 1.2), # Ajustar tamaño si es necesario
                  # Si usas plot_tensor_images, los args serían diferentes, ej:
                  # plot_tensor_images(images_for_plot, normalize_method='none', figsize_per_image=(1.2, 1.2))
                  # Le decimos 'none' a la normalización interna porque ya lo hicimos.
              )
              plt.suptitle("Imágenes Generadas ") # Añadir título
              plt.show() # Mostrar la figura
          except NameError:
              print("Error: La función de ploteo ('plot_image_grid' o 'plot_tensor_images') no está definida.")
              print("Asegúrate de haberla definido o importado previamente.")
          except Exception as e:
              print(f"Ocurrió un error durante la visualización: {e}")

      else:
          # Esto no debería pasar si generated_images_fid no está vacío, pero por si acaso
          print("No hay imágenes generadas para mostrar (num_to_plot <= 0).")
  elif 'generated_images_fid' not in locals():
      print("La variable 'generated_images_fid' no existe. No se pueden visualizar imágenes.")
  else:
      # Caso donde existe pero está vacía o no es un tensor
      print("La variable 'generated_images_fid' está vacía o no es un Tensor. No se pueden visualizar imágenes.")
      
      
      
def is_metric(sde_instance, score_model, T, DEVICE, BATCH_SIZE, IS_SPLITS, t_final=1e-4, n_images_for_is=200, steps=500):
  generation_device = DEVICE # El dispositivo donde corre tu modelo/sampler
  image_shape = (3, 32, 32) # Para MNIST

  # 1) Cálculo de sigma^2(T) con la fórmula dada
  sigma_cuadrado_evaluado_en_T = sde_instance.sigma_t(T) ** 2

  # 2) Creación de un tensor de ruido con media 0 y varianza 1
  #    El shape (n_images, 3, 32, 32) representa un lote de n_images de 3 canales 32x32
  noise = torch.randn(n_images_for_is, 3, 32, 32, device=DEVICE) # Changed to match CIFAR-10 image dimensions

  # 3) Escalar el ruido para que tenga varianza sigma_cuadrado_evaluado_en_T
  image_T = noise * np.sqrt(sigma_cuadrado_evaluado_en_T)
  #image_T

  print("Generating images for IS...")
  with torch.no_grad():
      times, synthetic_images_trajectory = euler_maruyama_integrator(
          image_T,
          t_0=T,
          t_end=t_final,
          n_steps=500,
          drift_coefficient=partial(
              sde_instance.backward_drift_coefficient,
              score_model=score_model,
          ),
          diffusion_coefficient=sde_instance.diffusion_coefficient,
      )

  # Extraer las imágenes finales
  final_generated_images = synthetic_images_trajectory[..., -1] # Shape: (n_images_for_is, C, H, W)
  print(f"Shape of final generated images: {final_generated_images.shape}")

  # --- Calcular Inception Score ---
  print(f"Calculating IS on device: {DEVICE}")


  mean_is, std_is = get_inception_score_for_generated_images(
      generated_images=final_generated_images.to(DEVICE), # Mover imágenes al dispositivo de IS
      batch_size=BATCH_SIZE,       # Ajusta según memoria de is_device
      n_splits=IS_SPLITS,         # Estándar, pero requiere suficientes imágenes
      device=DEVICE
  )

  display(Markdown(f"### Inception Score calculado:"))
  display(Markdown(f"### Media: {mean_is:.4f}"))
  display(Markdown(f"### Std Dev: {std_is:.4f}"))
  return (mean_is, std_is)