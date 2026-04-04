from data_parsing import parse_telemetry

# Отримуєш готові DataFrames
data = parse_telemetry("00000001.BIN")
df_gps = data['gps']
df_imu = data['imu']
print(df_gps)
print(df_imu)