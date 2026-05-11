# KUKA Pendulum RL (PPO) 🤖⚖️

> **🚧 Work in Progress:** This project is currently under active development. The custom robot model and the simulation environment are still being refined and expanded.

An applied Reinforcement Learning (RL) project focused on teaching a custom KUKA robot model to balance an inverted pendulum. The agent is trained using the **Proximal Policy Optimization (PPO)** algorithm within a custom OpenAI Gym environment.

---

## 🎯 About the Project

The core objective of this project is to develop a reliable control policy for a robotic arm to keep a pendulum perfectly balanced. By utilizing Reinforcement Learning, the robot learns through trial and error, continuously improving its balancing strategy based on the rewards it receives from the environment.

**Key Technologies & Concepts:**
*   **Reinforcement Learning (RL):** Training an agent to make sequential decisions.
*   **PPO Algorithm:** A robust and efficient policy gradient method used to train the balancing model.
*   **OpenAI Gym:** Used to create a custom, standardized environment for the robot to interact with.
*   **TensorBoard:** Utilized for tracking and visualizing training metrics (e.g., reward progression, loss).

---

## 📂 Project Structure

*   `main.py` — The main entry point of the project. Handles the initialization of the environment and the execution of the training or testing loops.
*   `openai_gym.py` — Contains the custom OpenAI Gym environment implementation specifically designed for the KUKA robot and the pendulum physics.
*   `kuka_balans_model.zip` — A pre-trained PPO model. It is bundled and ready to be loaded if you just want to watch the robot play/balance without retraining.
*   `kuka_tensorboard/` — Directory containing the log files generated during training. These can be visualized using TensorBoard to analyze the learning process.

---

## 🚀 Getting Started

*(Note: Setup instructions will be expanded as the project matures).*

### Prerequisites
Make sure you have the required Python libraries installed (e.g., `stable-baselines3`, `gymnasium`, `tensorboard`).

### Viewing the Pre-trained Model
To see the robot in action using the already trained model (`kuka_balans_model.zip`), you can run the main script (ensure the script is set to evaluation/render mode):
```bash
python main.py
