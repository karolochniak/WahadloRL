import sys
import numpy as np
import matplotlib.pyplot as plt
import math

from controller import Supervisor
from sympy import true

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit('Błąd importu. Uruchom: "pip install numpy gymnasium stable-baselines3"')


class PioneerSwingUpEnv(Supervisor, gym.Env):
    def __init__(self):
        super().__init__()

        self.x_threshold = 3.0
        self.max_steps = 500
        self.current_step = 0
        self.wind_force = 0.0

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
        self.wind_force = 0.0

        self.__wheels = []
        for name in ['left wheel', 'right wheel']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)

            # KOREKTA PRZYSPIESZENIA (zgodnie z wymogami)
            wheel.setAcceleration(60.0)
            self.__wheels.append(wheel)

        self.__sensor = self.getDevice('hinge sensor')
        if self.__sensor:
            self.__sensor.enable(self.__timestep)

        super().step(self.__timestep)

        return np.array([0, 0, 1.0, 0.0, 0], dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1

        # KOREKTA PRĘDKOŚCI MAKSYMALNEJ (ograniczenie do 20)
        speed = float(action[0]) * 20.0

        for wheel in self.__wheels:
            wheel.setVelocity(speed)

        endpoint = self.getFromDef("WAHADLO_SOLID")

        for _ in range(5):
            if self.wind_force != 0.0 and endpoint:
                endpoint.addForce([self.wind_force, 0.0, 0.0], False)
            super().step(self.__timestep)

        robot = self.getSelf()
        pos_x = robot.getPosition()[0]
        vel_x = robot.getVelocity()[0]
        angle = self.__sensor.getValue() if self.__sensor else 0.0
        ang_vel = endpoint.getVelocity()[4] if endpoint else 0.0

        state = np.array([
            pos_x, vel_x, np.cos(angle), np.sin(angle), ang_vel
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
        print("Rozpoczynamy trening...")
        model = PPO('MlpPolicy', env, verbose=2, device='cpu')
        model.learn(total_timesteps=1000000)
        model.save("pioneer_swingup_model_10rad")
    else:
        print("Wczytuję gotowy model...")
        model = PPO.load("pioneer_swingup_model_final.zip", env=env)

        obs, _ = env.reset()

        positions = []
        raw_angles_rad = []
        actions_list = []
        times = []
        current_step = 0

        ENABLE_NOISE = False
        ENABLE_WIND = False

        print(f"Start epizodu (Szum: {ENABLE_NOISE}, Wiatr: {ENABLE_WIND})...")

        while True:
            pos_x = obs[0]
            angle_rad = math.atan2(obs[3], obs[2])

            current_time = current_step * 0.04

            if ENABLE_WIND and 5.0 <= current_time <= 5.1:
                env.wind_force = 3.0
            else:
                env.wind_force = 0.0

            positions.append(pos_x)
            raw_angles_rad.append(angle_rad)
            times.append(current_time)

            noisy_obs = obs.copy()
            if ENABLE_NOISE:
                noise_angle_rad = np.random.normal(0.0, 0.04)
                noisy_angle = angle_rad + noise_angle_rad
                noisy_obs[2] = math.cos(noisy_angle)
                noisy_obs[3] = math.sin(noisy_angle)

            action, _ = model.predict(noisy_obs, deterministic=True)
            actions_list.append(float(action[0]))

            obs, reward, terminated, truncated, _ = env.step(action)
            current_step += 1

            if terminated or truncated:
                break

        unwrapped_angles_rad = np.unwrap(raw_angles_rad)
        angles_deg = np.degrees(unwrapped_angles_rad)
        positions = np.array(positions)
        actions = np.array(actions_list)

        target_angle = 180.0
        target_pos = 0.0
        rms_angle = np.sqrt(np.mean((angles_deg - target_angle) ** 2))
        rms_pos = np.sqrt(np.mean((positions - target_pos) ** 2))
        dt = 0.04
        energy_measure = np.sum(actions ** 2) * dt

        with open("metryki_PPO.txt", "w") as f:
            f.write(f"--- WYNIKI PPO (Szum: {ENABLE_NOISE}, Wiatr: {ENABLE_WIND}) ---\n")
            f.write(f"Czas trwania: {current_time:.2f} s\n")
            f.write(f"RMS bledu kata (wzgledem 180 deg): {rms_angle:.4f} stopni\n")
            f.write(f"RMS bledu pozycji (wzgledem 0 m): {rms_pos:.4f} m\n")
            f.write(f"Miara energii sterowania (calka u^2 dt): {energy_measure:.4f}\n")

        print("\n--- ZAPISANO METRYKI DO PLIKU: metryki_PPO.txt ---")
        print(f"RMS Kąta:    {rms_angle:.4f} stopni")
        print(f"RMS Pozycji: {rms_pos:.4f} m")
        print(f"Energia:     {energy_measure:.4f}\n")

        print("Rysowanie połączonego wykresu...")
        plt.style.use('default')
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')

        ax2 = ax1.twinx()
        line1, = ax1.plot(times, angles_deg, color='#005b82', linewidth=2.5, label='kąt')
        line2, = ax2.plot(times, positions, color='#e26b32', linewidth=2.5, label='pozycja')

        if ENABLE_WIND:
            ax1.axvspan(5.0, 5.1, color='#ffe082', alpha=0.3, label='Podmuch wiatru 2N')

        ax1.set_title('Regulacja PPO', color='black', pad=15, fontweight='bold', fontsize=14)
        ax1.set_xlabel('Czas symulacji [s]', color='black', fontsize=11)
        ax1.set_ylabel('Pozycja kątowa wahadła [°]', color='black', fontsize=11)
        ax2.set_ylabel('Pozycja wózka [m]', color='black', fontsize=11)

        ax1.grid(True, color='#d3d3d3', linestyle='-', linewidth=0.7)

        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', frameon=True, edgecolor='#d3d3d3')
        ax1.tick_params(colors='black')
        ax2.tick_params(colors='black')
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        plt.tight_layout()
        plot_filename = 'wykres_PPO.png'
        plt.savefig(plot_filename, dpi=300, facecolor='white', edgecolor='none')
        print(f"Zapisano wykres jako '{plot_filename}'!")

        env.close()
        sys.exit(0)

if __name__ == '__main__':
    main()
