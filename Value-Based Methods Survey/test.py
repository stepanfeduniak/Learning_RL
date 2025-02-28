import gymnasium as gym
import ale_py
import imageio
gym.register_envs(ale_py)


def record_breakout_game(max_steps=500, output_file="breakout.gif"):
    # Create the Breakout environment with rgb_array render mode.
    # This mode makes env.render() return an image frame.
    env = gym.make("Breakout-v4", render_mode="rgb_array")
    
    frames = []
    observation, info = env.reset()
    # Capture the initial frame.
    frames.append(env.render())
    
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        # Use a random action for demonstration purposes.
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if steps==10:
            print(observation)
        # Append the current frame.
        frames.append(env.render())
        steps += 1

    env.close()
    
    # Save all frames as a GIF.
    imageio.mimsave(output_file, frames, fps=30)
    print(f"Saved game recording to {output_file}")

if __name__ == "__main__":
    record_breakout_game()
