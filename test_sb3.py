import gym
import time
import torch
# from gym.wrappers import Monitor
import phoenix_drone_simulation
# from gym import wrappers
from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv



def test_with_sb3(env_id='DroneHoverBulletEnvWithAdversary-v0', alg='PPO', path=None):
    # register and initilaize the environment
    # assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
    # register(id=env_id, entry_point="{}:{}".format(
    #         DroneHoverBulletEnvWithAdversary.__module__, 
    #         DroneHoverBulletEnvWithAdversary.__name__), max_episode_steps=500)
    env = gym.make(env_id) 
    # env = wrappers.Monitor(env, '/localhome/hha160/projects/quadrotor-stabilization/sb3_test_videos', force=True)

    # Setup sb3 algorithm model and load the trained model
    # TODO: future could tune more hyperparameters
    model_path = path
    if alg == 'PPO':
        model = PPO(policy='MlpPolicy', env=env, verbose=1)
        model = PPO.load(model_path)
    elif alg == 'SAC':
        model = SAC(policy='MlpPolicy', env=env, verbose=1)
        model = SAC.load(model_path)
    else:
        print("Please check out the algorithm name again.")
    print("The model has been loaded successfully!")
    
    video_folder = '/localhome/hha160/projects/quadrotor-stabilization/test_videos'
    video_length = 100

    # env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x == 0, 
    #                        video_length=video_length, name_prefix=f"random-agent-{env_id}")
    
    env.render() 
    while True:
    # for _ in range(video_length+1):
        obs = env.reset()
        done = False
        while not done:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # env.render(mode="rgb_array")

            time.sleep(0.05)
            if done:
                obs = env.reset()


if __name__ == "__main__":
    # model_path = 'sb3_models/DroneHoverBulletEnvWithAdversary-v0/08_24_15_04/PPO_10000000.zip'
    model_path = 'sb3_models/DroneHoverBulletEnvWithAdversary-v0/08_24_16_21/PPO_10000000.zip'
    # model_path = 'runs/original_ppo/our_distb/DroneHoverBulletEnvWithAdversary-v0/ppo/2023-08-25__23-35-14/seed_40226/torch_save/model.pt'
    test_with_sb3(alg='PPO', path=model_path)
