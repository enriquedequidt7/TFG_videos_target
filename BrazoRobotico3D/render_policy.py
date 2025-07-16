import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import DDPG


from robotic_arm_env import RoboticArmEnv
from config import CONFIG


#renderiza la politica de comportamiento y guardarla en .gif
def render_policy_performance():
    env = RoboticArmEnv()
    try:
        model = DDPG.load(CONFIG["policy_path"], env=env)
    except FileNotFoundError:
        print(f"Error: Policy file '{CONFIG['policy_path']}' not found.")
        return
    
    episode_rewards = []
    
    for episode in range(CONFIG["num_episodes"]):
        state, _ = env.reset()
        total_reward = 0
        frames = []
        
        frame = env.render_frame()
        if frame is not None and isinstance(frame, np.ndarray) and frame.max() > 0:
            frames.append(frame.copy())
        
        for step in range(CONFIG["max_steps_per_episode"]):
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            
            frame = env.render_frame()
            if frame is not None and isinstance(frame, np.ndarray) and frame.max() > 0:
                frames.append(frame.copy())
            
            if terminated or truncated:
                print(f"Episode {episode + 1} terminated at step {step + 1} "
                      f"(done={terminated}, truncated={truncated})")
                break
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
        
        if frames:
            try:
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111)
                ax.axis('off')
                ims = [[ax.imshow(frame, animated=True)] for frame in frames]
                ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat=False)
                plt.title(f"3D Robotic Arm DDPG Episode {episode + 1}")
                gif_filename = f"{CONFIG['gif_prefix']}_{episode + 1}.gif"
                ani.save(gif_filename, writer='pillow')
                print(f"Saved animation as: {gif_filename}")
                plt.close()
            except Exception as e:
                print(f"Error saving GIF for episode {episode + 1}: {e}")
    
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, CONFIG["num_episodes"] + 1), episode_rewards, marker='o')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DDPG Policy Performance on 3D Robotic Arm")
        plt.grid(True)
        plt.savefig("robotic_arm_3d_performance.png")
        plt.close()
    except Exception as e:
        print(f"Error generating performance plot: {e}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average Reward over {CONFIG['num_episodes']} episodes: {avg_reward:.2f}")
    
    env.close()