# [Lorena & Jose, Generación de Imágenes con SDEs y Procesos de Difusión]

## Visión General

Este proyecto explora la generación de imágenes sintéticas mediante modelos de difusión basados en el formalismo de Ecuaciones Diferenciales Estocásticas (SDEs), siguiendo el marco teórico propuesto por Song et al. (2021) en "Score-Based Generative Modeling through Stochastic Differential Equations". El objetivo principal es implementar, analizar y evaluar un sistema modular para la generación de imágenes de alta fidelidad, investigando el impacto de diferentes formulaciones de SDEs, algoritmos de muestreo (_samplers_), arquitecturas de modelos de _score_, y técnicas de generación condicional e imputación.

El código está implementado principalmente en Python utilizando PyTorch. Se proporcionan cuadernos de Jupyter (`notebooks`) para facilitar la experimentación, la visualización de resultados y el cálculo de métricas de evaluación.

## Características Principales

* **Múltiples Formulaciones de SDEs:**
    * Variance Exploding SDE (VE-SDE)
    * Variance Preserving SDE (VP-SDE) con _schedules_ de ruido lineal y cosenoidal.
    * Sub-Variance Preserving SDE (SubVP-SDE) con _schedules_ de ruido lineal y cosenoidal.
* **Diversos Algoritmos de Muestreo (_Samplers_):**
    * Integrador de Euler-Maruyama.
    * Muestreadores de Predictor-Corrector (PC).
    * Solucionadores para la Probability Flow ODE (incluyendo un integrador exponencial de Euler).
* **Generación Condicional por Clase:**
    * Implementación de guiado por clasificador utilizando un modelo `TimeDependentWideResNet`.
    * Muestreador condicional Euler-Maruyama.
* **Imputación de Imágenes:**
    * Capacidad para rellenar regiones faltantes en imágenes utilizando un _sampler_ de imputación dedicado.
* **Modelos de _Score_ Flexibles:**
    * Arquitecturas basadas en U-Net (incluyendo una similar a NCSN++) para la estimación del _score_ $\nabla_x \log p_t(x)$.
    * Embeddings de tiempo mediante _Gaussian Random Fourier Features_.
* **Métricas de Evaluación Estándar:**
    * Fréchet Inception Distance (FID).
    * Inception Score (IS).
    * Bits Per Dimension (BPD) / Negative Log-Likelihood (NLL) vía Probability Flow ODE.
* **Cuadernos Interactivos:** Jupyter notebooks para configuración de experimentos, generación, visualización y evaluación.
* **Modelos Pre-entrenados:** Se proporcionan (o se enlaza a) modelos de _score_ y clasificadores pre-entrenados para facilitar la experimentación sin necesidad de largos procesos de entrenamiento.

## Fundamentos Teóricos (Breve)

El enfoque se basa en modelar la generación de datos como la inversión de un proceso de difusión definido por una SDE:

1.  **Proceso Directo (Forward SDE):** Una SDE predefinida $dx = f(x,t)dt + g(t)dW(t)$ transforma gradualmente una muestra de datos $x(0)$ en ruido $x(T)$ (distribución prior) a lo largo del tiempo $t \in [0, T]$.
2.  **Proceso Inverso (Reverse-Time SDE):** Existe una SDE inversa $dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{W}(t)$ que, si se conoce el _score_ $\nabla_x \log p_t(x)$ de la distribución de datos perturbados en cada instante $t$, puede transformar muestras de la distribución prior $x(T)$ de nuevo en muestras de la distribución de datos $x(0)$.
3.  **Estimación del Score:** Un modelo de red neuronal $s_\theta(x,t)$ se entrena para aproximar $\nabla_x \log p_t(x)$ mediante técnicas de _score matching_.
4.  **Probability Flow ODE:** Una ODE determinista $dx = [f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)]dt$ comparte las mismas marginales que la SDE y permite el cálculo exacto de la verosimilitud.

Para más detalles teóricos, se recomienda consultar [Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021](https://arxiv.org/abs/2011.13456).

## Estructura del Proyecto

El repositorio está organizado en los siguientes directorios principales:

* `diffusion/`: Implementaciones de los procesos de difusión SDE (`sde.py`) y los _schedules_ de ruido (`schedules.py`).
* `models/`: Arquitecturas de los modelos de _score_ (`base_model.py`, `score_model.py`, `score_net.py`) y el clasificador dependiente del tiempo (`classifier.py`, conteniendo `TimeDependentWideResNet`).
* `samplers/`: Algoritmos de muestreo para generación incondicional (`euler_maruyama.py`, `predictor_corrector.py`, `probability_flow_ode.py`, `exponential_integrator.py`).
* `conditional_samplers/`: Algoritmos de muestreo para generación condicional (`euler_maruyama_conditional_class.py`, etc.).
* `metrics/`: Implementación de las métricas de evaluación (`bpd.py`, `fid.py`, `inception_score.py`).
* `imputation/`: Lógica para la imputación de imágenes (`imputation.py`).
* `utils/`: Funciones de utilidad, especialmente para visualización (`diffusion_utilities.py`).
* `notebooks/`: Cuadernos de Jupyter para la experimentación (e.g., `Generacion_Incondicional_CIFAR10.ipynb`, `Generacion_Condicional_CIFAR10.ipynb`, `Imputacion_Imagenes.ipynb`).
* `scripts/`: (Opcional) Scripts para entrenamiento de modelos, ejecución batch de evaluaciones, etc.
* `checkpoints/`: Directorio para guardar/cargar los modelos pre-entrenados.
* `data/`: (Opcional) Para almacenar datasets pequeños o metadatos.
* `tests/`: Pruebas unitarias y de integración implementadas con `pytest`.

## Configuración e Instalación

### Prerrequisitos

* Python 3.8+
* PyTorch (versión X.Y.Z, idealmente con soporte CUDA si se dispone de GPU)
* NumPy
* SciPy
* Matplotlib
* Seaborn (para algunos gráficos de visualización de métricas)
* torchvision
* tqdm
* scikit-image (para algunas métricas o utilidades)
* (Opcional) Otras librerías específicas que hayas usado.

### Pasos de Instalación

1.  Clona este repositorio:
    ```bash
    git clone [https://www.youtube.com/watch?v=KrJwqsuhZ8U](https://www.youtube.com/watch?v=KrJwqsuhZ8U)
    cd [Nombre de tu repositorio]
    ```
2.  Se recomienda crear un entorno virtual:
    ```bash
    python -m venv env
    source env/bin/activate  # En Linux/macOS
    # env\Scripts\activate  # En Windows
    ```
3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
    *(Deberás crear un archivo `requirements.txt` con todas las librerías y sus versiones, e.g., `torch==X.Y.Z`, `torchvision==A.B.C`, `numpy`, etc.)*

### Conjuntos de Datos

* **CIFAR-10:** Se descarga automáticamente a través de `torchvision.datasets.CIFAR10`.
* **MNIST:** Se descarga automáticamente a través de `torchvision.datasets.MNIST`.
* (Si usas otros datasets, indica cómo obtenerlos).

### Modelos Pre-entrenados

Se proporcionan modelos de _score_ y clasificadores pre-entrenados para facilitar la experimentación sin largos tiempos de entrenamiento. Estos se encuentran en el directorio `checkpoints/` o pueden ser descargados desde [Enlace a tus modelos pre-entrenados, si los alojas externamente]. Los cuadernos de Jupyter están configurados para cargar estos modelos por defecto.

## Uso

La forma principal de interactuar con el proyecto es a través de los cuadernos de Jupyter ubicados en el directorio `notebooks/`.

### Estructura General de los Cuadernos

1.  **Configuración Inicial:** Las primeras celdas permiten definir hiperparámetros globales (dataset, tipo de SDE, modelo de _score_ a cargar, dispositivo CPU/GPU).
2.  **Selección del Sampler:** Se puede elegir el algoritmo de muestreo y sus parámetros específicos (número de pasos, etc.).
3.  **Generación de Muestras:** Se ejecuta el proceso de generación.
4.  **Visualización:** Se muestran las imágenes generadas y, en muchos casos, la evolución del proceso de muestreo.
5.  **Evaluación (Opcional):** Celdas para calcular métricas (FID, IS, BPD) sobre las muestras generadas.

### Ejemplos de Uso

* **Generación Incondicional:**
    * Abrir `notebooks/Generacion_Incondicional_CIFAR10.ipynb` (o el análogo para MNIST).
    * Configurar el tipo de SDE (VE, VP-Lineal, VP-Cosenoidal, SubVP-Lineal, SubVP-Cosenoidal) y cargar el modelo de _score_ correspondiente.
    * Seleccionar uno o varios _samplers_ (Euler-Maruyama, Predictor-Corrector, etc.) y sus parámetros.
    * Ejecutar las celdas para generar imágenes y visualizar la evolución.
    * Ejecutar las celdas de métricas para una evaluación cuantitativa.

* **Generación Condicional por Clase:**
    * Abrir `notebooks/Generacion_Condicional_CIFAR10.ipynb`.
    * Seleccionar el modelo SDE base (VE, VP-Lineal, SubVP-Lineal), lo que cargará el modelo de _score_ incondicional y el clasificador `TimeDependentWideResNet` entrenado específicamente para esa SDE.
    * Elegir la clase objetivo de CIFAR-10 y el número de muestras.
    * Configurar el `ConditionalEulerMaruyamaSampler` (e.g., `guidance_scale`).
    * Ejecutar para generar imágenes de la clase seleccionada.

* **Imputación de Imágenes:**
    * Abrir `notebooks/Imputacion_Imagenes.ipynb`.
    * Cargar una imagen y una máscara (se proporcionan ejemplos con MNIST sobre CIFAR-10).
    * Seleccionar el modelo de _score_ y configurar el `imputation_sampler`.
    * Ejecutar para rellenar las regiones enmascaradas.

## Componentes Implementados

### SDEs
* `VESDE` (Variance Exploding)
* `VPSDE` (Variance Preserving)
* `SubVPSDE` (Sub-Variance Preserving)

### Schedules de Ruido
* `LinearSchedule`
* `CosineSchedule`

### Arquitecturas de Modelo de Score
* `ScoreNet` (basada en U-Net, estilo NCSN++ adaptable) con bloques como `ResidualBlock`, `SelfAttentionBlock`, `Downsample`, `Upsample`.
* Embeddings de tiempo `GaussianRandomFourierFeatures`.

### Arquitectura del Clasificador
* `TimeDependentWideResNet` (WRN-28-10 modificada para dependencia temporal).

### Samplers (Incondicionales y Condicionales)
* Euler-Maruyama (incondicional y condicional)
* Predictor-Corrector (incondicional y condicional VE, VP/SubVP)
* Probability Flow ODE Integrator (incondicional)
* Exponential Euler ODE Sampler (incondicional)
* Imputation Sampler

### Métricas
* Bits Per Dimension (BPD) y Negative Log-Likelihood (NLL) vía ODE
* Fréchet Inception Distance (FID)
* Inception Score (IS)

## Resultados de Ejemplo

(Esta sección es opcional para el README, puedes enlazar a la memoria o a los notebooks)

Las experimentaciones indican (para CIFAR-10):
* El modelo **SubVP-SDE Lineal** tiende a ofrecer los mejores resultados en métricas perceptuales como FID e IS.
* Los modelos **SubVP-SDE Cosenoidal** y **VP-SDE Cosenoidal** destacan en BPD (verosimilitud).
* Existe un _trade-off_ entre optimizar la verosimilitud y la calidad perceptual.
* La generación condicional permite dirigir la síntesis hacia clases específicas con éxito variable según la SDE y el clasificador.

Se recomienda consultar los cuadernos de Jupyter y la memoria del proyecto para un análisis detallado de los resultados y más ejemplos visuales.

## Pruebas

El proyecto incluye un conjunto de pruebas unitarias y de integración desarrolladas con `pytest`. Para ejecutar las pruebas, desde el directorio raíz del proyecto:
```bash
pytest
