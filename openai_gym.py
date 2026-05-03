import sys
import numpy as np
from controller import Supervisor

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Błąd importu. Uruchom: "pip install numpy gymnasium stable-baselines3"'
    )


class KukaBalancingEnv(Supervisor, gym.Env):
    def __init__(self):
        super().__init__()

        # Parametry fizyczne środowiska (zgodnie z OpenAI Gym CartPole)
        self.theta_threshold_radians = 0.2  # Max wychylenie wahadła (~12 stopni)
        self.x_threshold = 0.4  # Max dystans od środka areny

        # Obserwacje: [pozycja_x, prędkość_x, kąt_wahadła, prędkość_kątowa]
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)

        self.action_space = gym.spaces.Discrete(2)  # 0: Lewo (Tył), 1: Prawo (Przód)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.__timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        self.__sensor = None

    def reset(self, seed=None, options=None):
        # Reset symulacji Webots
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Inicjalizacja silników kół mecanum (wheel1-wheel4)
        self.__wheels = []
        for name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)

        # Inicjalizacja czujnika wahadła
        self.__sensor = self.getDevice('pole_2_sensor')
        if self.__sensor is None:
            self.__sensor = self.getDevice('position sensor')  # fallback
        self.__sensor.enable(self.__timestep)

        super().step(self.__timestep)
        return np.array([0, 0, 0, 0], dtype=np.float32), {}

    def step(self, action):
        # 1. Wykonanie akcji - Jazda przód/tył (wyższa prędkość dla KUKA)
        # Zgodnie z teorią kół mecanum, wszystkie koła w jedną stronę to ruch liniowy
        speed = 5.0 if action == 1 else -5.0
        for wheel in self.__wheels:
            wheel.setVelocity(speed)

        super().step(self.__timestep)

        # 2. Pobieranie stanu z symulatora
        robot = self.getSelf()
        endpoint = self.getFromDef("POLE_ENDPOINT")

        # Pozycja i prędkość liniowa robota
        pos_x = robot.getPosition()[0]
        vel_x = robot.getVelocity()[0]

        # Kąt i prędkość kątowa wahadła
        angle = self.__sensor.getValue()
        # Wyciągamy prędkość obrotową (oś Y/Z w zależności od orientacji)
        ang_vel = endpoint.getVelocity()[4]

        state = np.array([pos_x, vel_x, angle, ang_vel], dtype=np.float32)

        # 3. Sprawdzenie warunków zakończenia
        terminated = bool(
            pos_x < -self.x_threshold or
            pos_x > self.x_threshold or
            angle < -self.theta_threshold_radians or
            angle > self.theta_threshold_radians
        )

        # Nagroda: +1 za każdą chwilę utrzymania pionu
        reward = 1.0 if not terminated else 0.0

        return state, reward, terminated, False, {}

    def wait_keyboard(self):
        print("Czekam na kliknięcie w oknie Webots i wciśnięcie 'Y'...")
        keyboard = self.getKeyboard()
        keyboard.enable(self.__timestep)
        while keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)


def main():
    # 1. Stworzenie środowiska
    env = KukaBalancingEnv()
    check_env(env)

    # GŁÓWNY PRZEŁĄCZNIK
    # True = Trenuje od zera i zapisuje wynik.
    # False = Pomija trening, wczytuje plik .zip i od razu jedzie!
    TRAIN_MODE = False

    if TRAIN_MODE:
        print("Rozpoczynam trening...")
        model = PPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log="./kuka_tensorboard/")
        model.learn(total_timesteps=80000)

        # Zapisz "mózg" robota do pliku ZIP w folderze projektu
        model.save("kuka_balans_model")
        print("Trening zakończony! Model zapisany do pliku kuka_balans_model.zip.")
    else:
        print("Wczytuję zapisany model...")
        # Wczytaj gotowy "mózg"
        model = PPO.load("kuka_balans_model", env=env)

    # 3. Testowanie modelu (Działa zawsze po treningu lub po wczytaniu)
    print("Zaczynamy pokaz...")
    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == '__main__':
    main()

