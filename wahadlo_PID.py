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
r_wheel = 0.0975 # Promień koła łazika [m] 

E_ref = 2.0 * m * g * l
MAX_ACCEL = 10.0/r_wheel       
MAX_MOTOR_SPEED = 20   

# ==========================================
# 3. ZMIENNE STANU I NASTAWY
# ==========================================
prev_theta = 0.0
prev_error_theta = 0.0
target_velocity = 0.0
swing_up_done = False
integral_theta = 0.0
prev_robot_pos = 0.0

Kp_theta = 300.0
Kd_theta = 70.0
Ki_theta = 20.0*0

Kp_pos = 0.05
Kd_pos = 0.1

target_position = 0.0

sum_sq_error_theta = 0.0
sum_sq_error_pos = 0.0
integral_control_signal = 0.0
pid_steps_count = 0
metrics_printed = False

# --- KONFIGURACJA ZAPISU DO CSV ---
time_elapsed = 0.0
csv_file = open('dane_wahadla.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

csv_writer.writerow(['Czas_s', 'Kat_theta_rad', 'Pozycja_wozka_m']) 

print("SWING-UP")
robot_pos = 0

# ==========================================
# 4. GŁÓWNA PĘTLA SYMULACJI
# ==========================================
while robot.step(timestep) != -1:
    time_elapsed += dt
 
    # --- ODCZYT SENSORA KĄTA ---
    theta = position_sensor.getValue() - 0.03845
    omega = (theta - prev_theta) / dt
    prev_theta = theta   

    # --- ODCZYT POZYCJI I PRĘDKOŚCI ŁAZIKA ---
    robot_pos = ((left_encoder.getValue() + right_encoder.getValue()) / 2.0) * r_wheel
    robot_vel = (robot_pos - prev_robot_pos) / dt
    prev_robot_pos = robot_pos

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
            print("Wahadło przechwycone! Przełączam na stabilizację PID")
            swing_up_done = True
            prev_error_theta = math.atan2(math.sin(theta - math.pi), math.cos(theta - math.pi))

    else:
        # FAZA 2: STABILIZACJA PID + KASKADA POZYCJI
        error_pos = target_position - robot_pos
        target_error_theta = (Kp_pos * error_pos) - (Kd_pos * robot_vel)
        target_error_theta = max(-0.26, min(0.26, target_error_theta))
        target_angle = math.pi + target_error_theta

        error_theta = math.atan2(math.sin(theta - target_angle), math.cos(theta - target_angle))

        P_out = Kp_theta * error_theta

        integral_theta += error_theta * dt
        integral_theta = max(-5.0, min(5.0, integral_theta))
        I_out = Ki_theta * integral_theta
        
        derivative_theta = (error_theta - prev_error_theta) / dt
        D_out = Kd_theta * derivative_theta

        prev_error_theta = error_theta

        commanded_acceleration = P_out + I_out + D_out

        abs_error_theta = math.atan2(math.sin(theta - math.pi), math.cos(theta - math.pi))
        sum_sq_error_theta += abs_error_theta**2
        sum_sq_error_pos += error_pos**2
        pid_steps_count += 1

    commanded_acceleration = max(-MAX_ACCEL, min(MAX_ACCEL, commanded_acceleration))

    integral_control_signal += abs(commanded_acceleration)*r_wheel * dt
    target_velocity += -commanded_acceleration * dt
    target_velocity = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, target_velocity))

    left_motor.setVelocity(-target_velocity)
    right_motor.setVelocity(-target_velocity)

    # ----------------------------------------------------
    # WYNIKI PO 20 SEKUNDACH
    # ----------------------------------------------------
    if time_elapsed >= 20.0 and not metrics_printed:
        if pid_steps_count > 0:
            rms_theta = math.sqrt(sum_sq_error_theta / pid_steps_count)
            rms_pos = math.sqrt(sum_sq_error_pos / pid_steps_count)
        else:
            rms_theta = 0.0
            rms_pos = 0.0
            
        print("\n" + "="*50)
        print("WYNIKI SYMULACJI PO 20 SEKUNDACH:")
        print(f"RMS błędu kąta (w fazie PID):      {rms_theta:.5f} rad")
        print(f"RMS błędu pozycji (w fazie PID):   {rms_pos:.5f} m")
        print(f"Całka z modułu sygnału (cała sym.): {integral_control_signal:.5f}")
        print("="*50 + "\n")
        metrics_printed = True

csv_file.close()
