# Inverted Pendulum on a Pioneer Robot (PPO vs LQR vs PID)

Welcome to our project! We stabilized an inverted pendulum on a Pioneer 3-DX robot using Webots. To give ourselves something cool to compare in our final report, we implemented three different control approaches: classic PID, LQR, and Reinforcement Learning (PPO).

##  What's inside the files?

* **`wahadlo.wbt`** – The Webots world file (contains the physics, robot, and environment setup).
* **`PPORL.py`** – The RL agent script. It runs the simulation using our trained PPO model. It also handles generating plots.
* **`pioneer_swingup_model_final.zip`** – Our fully trained neural network model. **Important:** Do not unzip this file! The `stable-baselines3` library reads it straight from the `.zip`. Just keep it in the same directory as the script.
* **`wahadlo_lqr.py`** – The LQR controller script.
* **`wahadlo_PID.py`** – The classic PID controller script.

## How to run it
* Open the world: Launch Webots, go to File -> Open World... and select the wahadlo.wbt file.
* Pause it: Immediately pause the simulation in Webots so the robot doesn't fall over or drive away before your code connects.
* Run the controller:
* The easy way (extern): Make sure the robot's controller field in the Webots node tree is set to <extern>. Then, open your IDE (like PyCharm) or terminal and simply run the script you want to test (e.g., python PPORL.py).
* The built-in way: Click on the robot node in Webots, change the controller field to the script you want to use, and hit "Play" at the top of the Webots window.
* Check the plots: If you're running the PPORL.py script, let the episode finish. The code will automatically generate and save some clean, light-themed trajectory plots right in your project folder.
## Prerequisites

Before you run this, make sure you have Webots installed and a Python environment ready (e.g., via PyCharm). You just need to install a few packages in your terminal:

```bash
pip install numpy matplotlib gymnasium stable-baselines3
