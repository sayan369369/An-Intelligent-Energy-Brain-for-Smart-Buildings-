import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from smart_hvac_env import SmartHVACEnv  

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = tqdm(total=total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

TOTAL_TIMESTEPS = 300_000  
MODEL_SAVE_PATH = "dqn_smart_hvac_all_improvements" 
TENSORBOARD_LOGDIR = "./tb_logs"

def train_and_run():
    if not os.path.exists(TENSORBOARD_LOGDIR):
        os.makedirs(TENSORBOARD_LOGDIR)
    

    env = Monitor(SmartHVACEnv())
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,       
        buffer_size=100_000,      
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,               
        exploration_fraction=0.2, 
        verbose=1,
        tensorboard_log=TENSORBOARD_LOGDIR
    )
    
    print(f"Starting training for {TOTAL_TIMESTEPS} steps...")
    print("   - Physics: Seasons enabled, Strong AC (-8.5kW)")
    print("   - Goal: Learn to manage Winter vs Summer automatically")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=ProgressCallback(TOTAL_TIMESTEPS)
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"Saved new seasonal brain to {MODEL_SAVE_PATH}.zip")
    print("   - You can now run 'streamlit run gui/app.py'")

if __name__ == "__main__":
    train_and_run()