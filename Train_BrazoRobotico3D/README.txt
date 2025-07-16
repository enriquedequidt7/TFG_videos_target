
Este codigo implementa un sistema de aprendizaje por refuerzo (RL) para controlar un brazo robótico en 3D utilizando el algoritmo DDPG (Deep Deterministic Policy Gradient) de Stable Baselines3. El brazo se simula en un entorno personalizado de Gym (`RoboticArmEnv`), y se entrena un modelo de transición para predecir el siguiente estado dado el estado actual y la acción. El proyecto también genera un GIF que visualiza el movimiento del brazo.

Dependencias:
Para ejecutar este proyecto, necesitas los siguientes paquetes de Python:
- `numpy`
- `matplotlib`
- `imageio`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `collections`
- `random`

Instala las dependencias usando pip:

pip install numpy matplotlib imageio gymnasium stable-baselines3 torch


Estructura del Proyecto
El código está dividido en los siguientes archivos:
- `imports_and_config.py`: Contiene todas las importaciones y el diccionario de configuración (`CONFIG`) con los hiperparámetros.
- `robotic_arm_env.py`: Define la clase `RoboticArmEnv`, un entorno personalizado de Gym para el brazo robótico en 3D.
- `transition_model.py`: Define la clase `TransitionModel`, una red neuronal para predecir transiciones de estado.
- `data_and_training.py`: Incluye funciones para recolectar datos de transición y entrenar el modelo de transición.
- `main.py`: El script principal que entrena el modelo DDPG, recolecta datos, entrena el modelo de transición y genera un GIF.

Cómo Ejecutar:
1. Asegúrate de que todas las dependencias estén instaladas.

3. Ejecuta el script principal:
   python main.py
  
4. El script realizará lo siguiente:
   - Entrenará un modelo DDPG durante 100,000 pasos de tiempo.
   - Recolectará 20,000 tuplas de estado-acción-siguiente_estado usando acciones aleatorias.
   - Entrenará un modelo de transición durante 100 épocas.
   - Generará un GIF (`brazo_robotico_3d.gif`) que muestra el movimiento del brazo.
   - Guardará el modelo DDPG como `ddpg_robotic_arm.zip` y el modelo de transición como `transition_model.pth`.

Salidas
- `ddpg_robotic_arm.zip`: Modelo DDPG entrenado.
- `transition_model.pth`: Modelo de transición entrenado.
- `brazo_robotico_3d.gif`: Animación del brazo robótico alcanzando una altura objetivo.

Notas
- El brazo robótico se simula en un entorno 3D con enfoque en el movimiento vertical (eje z).
- El modelo DDPG utiliza ruido en las acciones para mejorar la exploración durante el entrenamiento.
- El modelo de transición se entrena para predecir el siguiente estado, lo que puede usarse para RL basado en modelos o análisis.
- La visualización en GIF muestra la base, el eslabón y dos segmentos del brazo, con una línea discontinua roja que indica la altura objetivo.

Requisitos
- Python 3.7 o superior.
- Un sistema con suficiente memoria y CPU/GPU para entrenar los modelos (se recomienda GPU para un entrenamiento más rápido).

Licencia
Este proyecto se proporciona tal cual para fines educativos. Siéntete libre de modificarlo y extenderlo según sea necesario.
