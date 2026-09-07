import sys
import numpy as np
import matplotlib.pyplot as plt
import math

from controller import Supervisor

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit('Bład importu. Uruchom: "pip install numpy gymnasium stable-baselines3"')


class PioneerSwingUpEnv(Supervisor, gym.Env):
    def __init__(self):
        super().__init__()

        self.x_threshold = 3.0
        self.max_steps = 500
        self.current_step = 0

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            1.0,
            1.0,
            np.finfo(np.float32).max
        ], dtype=np.float32)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.__timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        self.__sensor = None

    def reset(self, seed=None, options=None):
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        self.current_step = 0

        self.__wheels = []
        for name in ['left wheel', 'right wheel']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            wheel.setAcceleration(60.0)
            self.__wheels.append(wheel)

        self.__sensor = self.getDevice('hinge sensor')
        if self.__sensor:
            self.__sensor.enable(self.__timestep)

        super().step(self.__timestep)

        return np.array([0, 0, 1.0, 0.0, 0], dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        speed = float(action[0]) * 20.0

        for wheel in self.__wheels:
            wheel.setVelocity(speed)

        for _ in range(5):
            super().step(self.__timestep)

        robot = self.getSelf()
        endpoint = self.getFromDef("WAHADLO_SOLID")
        pos_x = robot.getPosition()[0]
        vel_x = robot.getVelocity()[0]
        angle = self.__sensor.getValue() if self.__sensor else 0.0
        ang_vel = endpoint.getVelocity()[4] if endpoint else 0.0

        state = np.array([
            pos_x,
            vel_x,
            np.cos(angle),
            np.sin(angle),
            ang_vel
        ], dtype=np.float32)

        reward = float((1.0 - np.cos(angle)) / 2.0)
        reward -= 0.05 * abs(pos_x)
        reward -= 0.02 * (float(action[0]) ** 2)

        if np.cos(angle) > 0.9:
            reward -= 0.1 * abs(vel_x)
            reward -= 0.1 * abs(ang_vel)
        else:
            reward -= 0.01 * abs(vel_x)
            reward -= 0.01 * abs(ang_vel)

        terminated = bool(pos_x < -self.x_threshold or pos_x > self.x_threshold)
        truncated = bool(self.current_step >= self.max_steps)

        return state, reward, terminated, truncated, {}


def main():
    env = PioneerSwingUpEnv()
    check_env(env)

    TRAIN_MODE = False

    if TRAIN_MODE:
        model = PPO('MlpPolicy', env, verbose=2, device='cpu')
        model.learn(total_timesteps=1000000)
        model.save("pioneer_swingup_model_final")
    else:
        model = PPO.load("pioneer_swingup_model_final", env=env)
        obs, _ = env.reset()

        positions = []
        angles = []
        times = []
        current_step = 0

        while True:
            pos_x = obs[0]
            angle_rad = math.atan2(obs[3], obs[2])
            angle_deg = math.degrees(angle_rad)

            current_time = current_step * 0.04

            positions.append(pos_x)
            angles.append(angle_deg)
            times.append(current_time)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            current_step += 1

            if terminated or truncated:
                print(f"Koniec epizodu! Zabrano {current_step} próbek ({current_time:.2f} s).")
                break
                
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.patch.set_facecolor('white')

        ax1.set_facecolor('white')
        ax1.plot(times, positions, color='#1f77b4', linewidth=2.5)
        ax1.set_title('Pozycja wózka w trakcie stabilizacji', color='black', pad=10, fontweight='bold')
        ax1.set_ylabel('Pozycja [m]', color='black')
        ax1.grid(True, color='#e0e0e0', linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color('black')
        ax1.spines['left'].set_color('black')
        ax1.tick_params(colors='black')

        ax2.set_facecolor('white')
        ax2.plot(times, angles, color='#d62728', linewidth=2.5)
        ax2.set_title('Kąt wahadła w trakcie stabilizacji', color='black', pad=10, fontweight='bold')
        ax2.set_xlabel('Czas [s]', color='black')
        ax2.set_ylabel('Kąt [stopnie]', color='black')
        ax2.grid(True, color='#e0e0e0', linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_color('black')
        ax2.spines['left'].set_color('black')
        ax2.tick_params(colors='black')

        plt.tight_layout(pad=2.0)
        plt.savefig('trajektoria_lotu_czas_jasny.png', dpi=300, facecolor='white', edgecolor='none',
                    bbox_inches='tight')
        print("Zapisano jasny wykres 'trajektoria_lotu_czas_jasny.png'!")

        env.close()
        import sys
        sys.exit(0)
if __name__ == '__main__':
    main()
