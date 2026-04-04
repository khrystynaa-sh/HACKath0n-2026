import numpy as np
import pandas as pd
# Імпортуємо твій парсер (заміни 'data_convert' на назву твого файлу з парсером, якщо вона інша)
from data_convert import parse_telemetry 

def calculate_haversine_distance(df_gps):
    """Обчислення загальної дистанції за формулою Haversine"""
    R = 6371000  # Радіус Землі в метрах

    # Переводимо в радіани
    lat = np.radians(df_gps['Lat'])
    lon = np.radians(df_gps['Lng'])

    # Зміщуємо масиви для знаходження різниці між сусідніми точками
    dlat = lat.shift(-1) - lat
    dlon = lon.shift(-1) - lon

    # Формула Haversine
    a = np.sin(dlat / 2)**2 + np.cos(lat) * np.cos(lat.shift(-1)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = R * c
    return distances.sum()

def calculate_imu_dynamics(df_imu):
    """Метод трапецій для інтегрування прискорень у швидкість"""
    # Крок часу в секундах
    dt = df_imu['time_s'].diff().fillna(0)

    # Віднімання гравітації (беремо середнє перших 50 точок, коли дрон стоїть на землі)
    # Це вирішує проблему нахиленого дрона на старті
    acc_x_clean = df_imu['AccX'] - df_imu['AccX'].iloc[0:50].mean()
    acc_y_clean = df_imu['AccY'] - df_imu['AccY'].iloc[0:50].mean()
    acc_z_clean = df_imu['AccZ'] - df_imu['AccZ'].iloc[0:50].mean()

    # Інтегруємо методом трапецій: V = sum( (a_поточне + a_попереднє)/2 * dt )
    vel_x = ((acc_x_clean + acc_x_clean.shift(1).fillna(0)) / 2 * dt).cumsum()
    vel_y = ((acc_y_clean + acc_y_clean.shift(1).fillna(0)) / 2 * dt).cumsum()
    vel_z = ((acc_z_clean + acc_z_clean.shift(1).fillna(0)) / 2 * dt).cumsum()

    # Горизонтальна швидкість (вектор X та Y)
    horizontal_speed = np.sqrt(vel_x**2 + vel_y**2)
    # Вертикальна швидкість (модуль Z)
    vertical_speed = np.abs(vel_z)

    # Максимальне прискорення (чисте, без гравітації)
    total_accel = np.sqrt(acc_x_clean**2 + acc_y_clean**2 + acc_z_clean**2)

    return {
        'max_horizontal_speed': horizontal_speed.max(),
        'max_vertical_speed': vertical_speed.max(),
        'max_acceleration': total_accel.max()
    }

def run_analytics(filename):
    """Головна функція, яка збирає всі метрики місії"""
    print(f"Читання та парсинг файлу {filename}...")

    # 1. Отримуємо розпарсені дані з твого парсера
    data = parse_telemetry(filename)
    df_gps = data['gps']
    df_imu = data['imu']

    # 2. Перевірка, чи є дані
    if df_gps.empty or df_imu.empty:
        print("Помилка: Немає даних GPS або IMU у файлі.")
        return

    # 3. Обчислення метрик
    total_distance = calculate_haversine_distance(df_gps)
    imu_metrics = calculate_imu_dynamics(df_imu)
    max_alt_gain = df_gps['Alt'].max() - df_gps['Alt'].min()
    duration = df_gps['time_s'].max() # або imu_df['time_s'].max()

    # 4. Вивід фінального звіту
    print("\n" + "="*50)
    print(" ПІДСУМКОВІ ПОКАЗНИКИ МІСІЇ (ЯДРО АНАЛІТИКИ)")
    print("="*50)
    print(f"Загальна пройдена дистанція:   {total_distance:.2f} метрів")
    print(f"Максимальна гориз. швидкість:  {imu_metrics['max_horizontal_speed']:.2f} м/с")
    print(f"Максимальна верт. швидкість:   {imu_metrics['max_vertical_speed']:.2f} м/с")
    print(f"Максимальне прискорення:       {imu_metrics['max_acceleration']:.2f} м/с²")
    print(f"Максимальний набір висоти:     {max_alt_gain:.2f} метрів")
    print(f"Загальна тривалість польоту:   {duration:.2f} секунд")
    print("="*50)

# Запуск скрипта
if __name__ == "__main__":
    # Встав правильний шлях до твого .BIN файлу
    FILE_PATH = r"00000001.BIN"
    run_analytics(FILE_PATH)
