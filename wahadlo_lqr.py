import math
import csv
import random
from controller import Supervisor

# ==========================================
# 1. INICJALIZACJA I KONFIGURACJA
# ==========================================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

# Silniki
left_motor = robot.getDevice("left wheel")
right_motor = robot.getDevice("right wheel")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Enkodery kół 
left_encoder = left_motor.getPositionSensor()
right_encoder = right_motor.getPositionSensor()
left_encoder.enable(timestep)
right_encoder.enable(timestep)

# Czujnik kątu wahadła
position_sensor = robot.getDevice("hinge sensor")
position_sensor.enable(timestep)


# ==========================================
# 2. PARAMETRY FIZYCZNE I GLOBALNE LIMITY
# ==========================================
m = 0.5     # Masa wahadła [kg]
g = 9.81    # Przyspieszenie ziemskie [m/s^2]
l = 0.306   # Odległość do środka masy wahadła [m]
r_wheel = 0.0975 # Promień koła łazika [m] (Upewnij się, że pasuje do modelu)

E_ref = 2.0 * m * g * l

# --- GLOBALNE OGRANICZENIA FIZYCZNE ---
MAX_ACCEL = 10.0       # Limit przyspieszenia (rad/s^2)
MAX_MOTOR_SPEED = 20   # Maksymalna prędkość kół (rad/s)


# ==========================================
# 3. ZMIENNE STANU I NASTAWY
# ==========================================
prev_theta = 0.0
prev_error_theta = 0.0
target_velocity = 0.0
swing_up_done = False
integral_theta = 0.0
target_position = 0
prev_robot_pos = 0.0

K_pos   = -3.1623
K_vel   = -7.7663
K_ang   = -100.2633
K_omega = -47.7742

sum_sq_error_theta = 0.0
sum_sq_error_pos = 0.0
integral_control_signal = 0.0
lqr_steps_count = 0
metrics_printed = False

# --- KONFIGURACJA ZAPISU DO CSV ---
time_elapsed = 0.0
csv_file = open('dane_wahadla_lqr.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Czas_s', 'Kat_theta_rad', 'Pozycja_wozka_m']) 

print("SWING-UP")

# ==========================================
# 4. GŁÓWNA PĘTLA SYMULACJI
# ==========================================
while robot.step(timestep) != -1:

    time_elapsed += dt
    
    # --- ODCZYT SENSORA KĄTA---
    theta = position_sensor.getValue() - 0.03845
    omega = (theta - prev_theta) / dt
    prev_theta = theta
    
    # --- ODCZYT POZYCJI I PRĘDKOŚCI ŁAZIKA ---
    robot_pos = ((left_encoder.getValue() + right_encoder.getValue()) / 2.0) * r_wheel
    robot_vel = (robot_pos - prev_robot_pos) / dt
    prev_robot_pos = robot_pos
    
    # --- ZAPIS DO PLIKU CSV ---
    csv_writer.writerow([time_elapsed, theta, robot_pos])
    
    commanded_acceleration = 0.0 
    
    
    if not swing_up_done:
        E_p = m * g * l * (1.0 - math.cos(theta))
        E_k = 0.5 * m * (l**2) * (omega**2)
        E = E_p + E_k
        
        if E < E_ref:
            trigger = omega * math.cos(theta)
            if trigger > 0:
                commanded_acceleration = MAX_ACCEL
            elif trigger < 0:
                commanded_acceleration = -MAX_ACCEL
        
        if abs(math.pi - abs(theta)) < 0.4:
            print("Wahadło przechwycone! Przełączam na stabilizację LQR...")
            swing_up_done = True
            target_position = robot_pos
            
    else:
        error_pos = robot_pos - target_position
        error_vel = robot_vel
        error_theta = math.atan2(math.sin(theta - math.pi), math.cos(theta - math.pi))
        error_omega = omega
        
        commanded_acceleration = -(K_pos * error_pos + K_vel * error_vel + K_ang * error_theta + K_omega * error_omega)

        sum_sq_error_theta += error_theta**2
        sum_sq_error_pos += error_pos**2
        lqr_steps_count += 1

    # ----------------------------------------------------
    # GLOBALNE OGRANICZENIA FIZYCZNE I CAŁKOWANIE
    # ----------------------------------------------------
    commanded_acceleration = max(-MAX_ACCEL, min(MAX_ACCEL, commanded_acceleration))

    integral_control_signal += abs(commanded_acceleration) * dt

    target_velocity += -commanded_acceleration * dt
    max_linear_speed = MAX_MOTOR_SPEED * r_wheel
    target_velocity = max(-max_linear_speed, min(max_linear_speed, target_velocity))
    
    target_omega = target_velocity / r_wheel
    left_motor.setVelocity(-target_omega)
    right_motor.setVelocity(-target_omega)
    
    # ----------------------------------------------------
    # WYNIKI PO 20 SEKUNDACH
    # ----------------------------------------------------
    if time_elapsed >= 20.0 and not metrics_printed:
        if lqr_steps_count > 0:
            rms_theta = math.sqrt(sum_sq_error_theta / lqr_steps_count)
            rms_pos = math.sqrt(sum_sq_error_pos / lqr_steps_count)
        else:
            rms_theta = 0.0
            rms_pos = 0.0
            
        print("\n" + "="*50)
        print("WYNIKI SYMULACJI PO 20 SEKUNDACH:")
        print(f"RMS błędu kąta (w fazie LQR):      {rms_theta:.5f} rad")
        print(f"RMS błędu pozycji (w fazie LQR):   {rms_pos:.5f} m")
        print(f"Całka z modułu sygnału (cała sym.): {integral_control_signal:.5f} m/s")
        print("="*50 + "\n")
        metrics_printed = True

csv_file.close()
