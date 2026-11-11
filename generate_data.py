
import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

with open("data/renewable_energy.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "date",
        "temperature",
        "solar_radiation",
        "wind_speed",
        "humidity",
        "cloud_cover",
        "solar_generation",
        "wind_generation",
        "total_generation"
    ])

    start_date = datetime(2024, 1, 1)

    for i in range(1500):
        date = start_date + timedelta(hours=i)

        temperature = round(random.uniform(15, 42), 2)
        solar_radiation = round(max(0, 800 * math.sin(math.pi * (date.hour - 6) / 12) + random.uniform(-50, 50)), 2)
        wind_speed = round(random.uniform(2, 25), 2)
        humidity = round(random.uniform(20, 95), 2)
        cloud_cover = round(random.uniform(0, 100), 2)

        solar_generation = round(
            solar_radiation * (1 - cloud_cover / 150) * 0.02, 2
        )

        wind_generation = round(wind_speed ** 2 * 0.08, 2)

        total_generation = round(
            solar_generation + wind_generation, 2
        )

        writer.writerow([
            date, temperature, solar_radiation,
            wind_speed, humidity, cloud_cover,
            solar_generation, wind_generation,
            total_generation
        ])

print("Renewable energy dataset created successfully!")
