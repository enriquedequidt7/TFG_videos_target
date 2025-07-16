#Diccionario de configuracion
CONFIG = {
    # DDPG Training Parameters
    "total_timesteps": 100000,
    "learning_rate": 1e-3,
    "buffer_size": 1000000,
    "batch_size": 128,
    "gamma": 0.99,
    "tau": 0.005,
    "action_noise_sigma": 0.3,
    "policy": "MlpPolicy",
    # Transition Model Parameters
    "transition_model_hidden_layers": [256, 256],
    "transition_model_lr": 1e-3,
    "transition_model_epochs": 100,
    "transition_model_batch_size": 128,
    "transition_data_size": 20000,
    "validation_split": 0.2,
}