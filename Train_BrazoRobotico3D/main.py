import numpy as np
import imageio
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  
import gymnasium as gym
from gymnasium import spaces

from imports_and_config import CONFIG
#clase que define el entorno del brazo robótico
class RoboticArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32)
        self.arm_length = 100
        self.max_angle = np.pi
        self.dt = 0.05
        self.state = None
        self.target_angle = 0.0
        self.step_count = 0
        self.max_steps = 300
        # visualización 3D
        self.base_radius = 50
        self.base_height = 10
        self.body_height = 50
        self.body_radius = 5
        self.joint_radius = 8
        self.arm_thickness = 10
        self.gripper_length = 20
        self.gripper_width = 5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0])
        self.target_angle = np.random.uniform(-self.max_angle, self.max_angle)
        self.step_count = 0
        return np.array([self.state[0], self.state[1], self.target_angle]), {}

    def step(self, action):
        angle, angular_velocity = self.state
        torque = np.clip(action[0], -1.0, 1.0)
        angular_acceleration = torque - 0.05 * angular_velocity
        angular_velocity += angular_acceleration * self.dt
        angle += angular_velocity * self.dt
        angle = np.clip(angle, -self.max_angle, self.max_angle)
        self.state = np.array([angle, angular_velocity])
        
        angle_error = abs(angle - self.target_angle)
        reward = -angle_error - 0.01 * abs(torque)
        if angle_error < 0.05:
            reward += 10.0
        
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        return np.array([angle, angular_velocity, self.target_angle]), reward, terminated, truncated, {}

    def render_frame(self):
        angle, _ = self.state
        # se calcula posiciones del brazo y del objetivo
        arm_end = [np.cos(angle) * self.arm_length, np.sin(angle) * self.arm_length, self.body_height]
        target_end = [np.cos(self.target_angle) * self.arm_length, np.sin(self.target_angle) * self.arm_length, self.body_height]

        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        #limites del mapa
        ax.set_xlim(-self.arm_length * 1.2, self.arm_length * 1.2)
        ax.set_ylim(-self.arm_length * 1.2, self.arm_length * 1.2)
        ax.set_zlim(0, self.body_height + self.arm_length * 1.2)
        ax.set_box_aspect([1, 1, 1])

        #dibujado

        u = np.linspace(0, 2 * np.pi, 20)
        z = np.linspace(0, self.base_height, 10)
        U, Z = np.meshgrid(u, z)
        x = self.base_radius * np.cos(U)
        y = self.base_radius * np.sin(U)
        ax.plot_surface(x, y, Z, color='k', alpha=0.8)


        x = self.body_radius * np.cos(U)
        y = self.body_radius * np.sin(U)
        ax.plot_surface(x, y, Z * (self.body_height / self.base_height), color='k', alpha=0.8)


        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        U, V = np.meshgrid(u, v)
        x = self.joint_radius * np.cos(U) * np.sin(V)
        y = self.joint_radius * np.sin(U) * np.sin(V)
        z = self.body_height + self.joint_radius * np.cos(V)
        ax.plot_surface(x, y, z, color='gray', alpha=0.8)


        def cuboid_data(o, size=(1, 1, 1)):
            l, w, h = size
            x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
            y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
                 [o[1], o[1], o[1] + w, o[1] + w, o[1]],
                 [o[1], o[1], o[1], o[1], o[1]],
                 [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
            z = [[o[2], o[2], o[2], o[2], o[2]],
                 [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
                 [o[2], o[2], o[2] + h, o[2] + h, o[2]],
                 [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
            return np.array(x), np.array(y), np.array(z)

        arm_vec = np.array(arm_end)
        arm_start = np.array([0, 0, self.body_height])
        arm_length_actual = np.linalg.norm(arm_vec - arm_start)
        arm_x, arm_y, arm_z = cuboid_data(arm_start, size=(arm_length_actual, self.arm_thickness, self.arm_thickness))
        direction = (arm_vec - arm_start) / arm_length_actual
        vertices = np.vstack([arm_x.flatten(), arm_y.flatten(), arm_z.flatten()]).T
        vertices = vertices @ np.array([
            [direction[0], -direction[1], 0],
            [direction[1], direction[0], 0],
            [0, 0, 1]
        ]).T
        arm_x = vertices[:, 0].reshape(arm_x.shape)
        arm_y = vertices[:, 1].reshape(arm_y.shape)
        arm_z = vertices[:, 2].reshape(arm_z.shape)
        ax.add_collection3d(Poly3DCollection([list(zip(arm_x[i], arm_y[i], arm_z[i])) for i in range(4)], color='b', alpha=0.8))


        gripper1_start = arm_end + np.array([-self.gripper_length/2, self.arm_thickness/2, 0])
        gripper2_start = arm_end + np.array([-self.gripper_length/2, -self.arm_thickness/2 - self.gripper_width, 0])
        gripper1_x, gripper1_y, gripper1_z = cuboid_data(gripper1_start, size=(self.gripper_length, self.gripper_width, self.gripper_width))
        gripper2_x, gripper2_y, gripper2_z = cuboid_data(gripper2_start, size=(self.gripper_length, self.gripper_width, self.gripper_width))
        ax.add_collection3d(Poly3DCollection([list(zip(gripper1_x[i], gripper1_y[i], gripper1_z[i])) for i in range(4)], color='gray', alpha=0.8))
        ax.add_collection3d(Poly3DCollection([list(zip(gripper2_x[i], gripper2_y[i], gripper2_z[i])) for i in range(4)], color='gray', alpha=0.8))


        ax.plot([0, target_end[0]], [0, target_end[1]], [self.body_height, target_end[2]], 'r--', linewidth=2, label='Target')

        #Etiquetas
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Angle: {angle:.2f} | Target: {self.target_angle:.2f}")
        ax.legend()

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return image

    def close(self):
        pass
    
def main():
    env = RoboticArmEnv()
    #rudio de acciones para DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=CONFIG["action_noise_sigma"] * np.ones(n_actions)
    )
    
    #Inicializar la instancia DDPG de stable baselines
    model = DDPG(
        policy=CONFIG["policy"],
        env=env,
        learning_rate=CONFIG["learning_rate"],
        buffer_size=CONFIG["buffer_size"],
        batch_size=CONFIG["batch_size"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"],
        action_noise=action_noise,
        verbose=1
    )

    # Entrena con DDPG
    model.learn(total_timesteps=CONFIG["total_timesteps"])
    model.save("ddpg_robotic_arm")
    
    # datos para el modelo de transicion
    print("Collecting data for transition model with random actions...")
    data = collect_transition_data(env, CONFIG["transition_data_size"])
    
    # Entrena transition model
    print("Training transition model...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    transition_model = train_transition_model(state_dim, action_dim, data, CONFIG)
    
    # Save transition model
    torch.save(transition_model.state_dict(), "transition_model.pth")
    
    # Generación de GIF
    frames = []
    obs, _ = env.reset()
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)  # Modo determinista para evaluación
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render_frame()
        frames.append(frame)
        if terminated or truncated:
            break

    # Guardar como GIF
    imageio.mimsave("brazo_robotico_3d.gif", frames, fps=30)
    print("GIF guardado como 'brazo_robotico_3d.gif'")
    print("Models saved as 'ddpg_robotic_arm.zip' and 'transition_model.pth'.")

if __name__ == "__main__":
    main()