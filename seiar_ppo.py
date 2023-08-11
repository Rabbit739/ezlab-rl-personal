import os
import hydra
from hydra.utils import instantiate
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback
)

sns.set_theme(style="whitegrid")

@hydra.main(version_base=None, config_path="conf", config_name="ppo_seiar")
def main(conf: DictConfig):
    train_env = instantiate(conf.seiar)
    check_env(train_env)
    log_dir = "./seiar_ppo_log"
    os.makedirs(log_dir, exist_ok=True)
    train_env = Monitor(train_env, log_dir)
    policy_kwargs = dict(
                            # activation_fn=torch.nn.ReLU,
                            # net_arch=[16, 32, 64, 16]
                        )
    model = PPO("MlpPolicy", train_env, verbose=0,
                policy_kwargs=policy_kwargs,
                clip_range=conf.clip_range,
                learning_rate=conf.lr)

    # callback setting
    eval_env = instantiate(conf.seiar)
    eval_callback = EvalCallback(
            eval_env,
            eval_freq=300*500,
            verbose=0,
            warn=False,
            log_path='eval_log',
            best_model_save_path='best_model'
        )
    checkpoint_callback = CheckpointCallback(save_freq=300*500, save_path='./checkpoints/',
                                             name_prefix='rl_model')
    callback = CallbackList([checkpoint_callback, eval_callback, ProgressBarCallback()])

    # Learning
    model.learn(total_timesteps=conf.n_steps, callback=callback)

    os.makedirs('figures', exist_ok=True)
    df = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
    sns.lineplot(data=df.r)
    plt.xlabel('episodes')
    plt.ylabel('The cummulative return')
    plt.savefig(f"figures/reward.png")
    plt.close()

    # Visualize Controlled SIR Dynamics
    model = PPO.load(f'best_model/best_model.zip')
    state = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, _, done, _ = eval_env.step(action)

    df = eval_env.dynamics
    best_reward = df.rewards.sum()
    df.to_excel("best_dynamics.xlsx")
    
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
    plt.suptitle(f"R = {df.rewards.sum():,.4f}")
    sns.lineplot(data=df, x='days', y='susceptible', color='r', ax=axes[0])

    sns.lineplot(data=df, x='days', y='infected', color='r', ax=axes[1])

    sns.lineplot(data=df, x='days', y='nus', color='k', drawstyle='steps-pre', ax=axes[2])
    axes[2].set_ylim([-conf.seiar.nu_daily_max*0.1, max(conf.seiar.nu_daily_max * 1.2, 0.01)])

    sns.lineplot(data=df, x='days', y='rewards', color='g', ax=axes[3])

    plt.subplots_adjust(hspace=0.25)

    plt.savefig(f"figures/best.png")
    plt.close()
    
    
    max_val = -float('inf')
    for path in os.listdir('checkpoints'):
        model = PPO.load(f'checkpoints/{path}')
        state = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, _, done, _ = eval_env.step(action)
        df = eval_env.dynamics

        cum_reward = df.rewards.sum()
        if cum_reward > max_val:
            max_val = cum_reward
            best_checkpoint = path

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
        plt.suptitle(f"R = {df.rewards.sum():,.4f}")
        sns.lineplot(data=df, x='days', y='susceptible', color='r', ax=axes[0])

        sns.lineplot(data=df, x='days', y='infected', color='r', ax=axes[1])

        sns.lineplot(data=df, x='days', y='nus', color='k', drawstyle='steps-pre', ax=axes[2])
        axes[2].set_ylim([-conf.seiar.nu_daily_max*0.1, max(conf.seiar.nu_daily_max * 1.2, 0.01)])

        sns.lineplot(data=df, x='days', y='rewards', color='g', ax=axes[3])

        plt.subplots_adjust(hspace=0.25)
        
        plt.savefig(f"figures/{path.replace('.zip', '.png')}")
        plt.close(fig)

        
        
if __name__ == '__main__':
    main()