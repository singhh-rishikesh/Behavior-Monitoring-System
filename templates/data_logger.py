import csv
import os

csv_file = "data/detection_data.csv"

if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Eye Movement", "Emotion", "Hand Gesture"])

def log_data(eye_direction, emotion, gesture):
    with open(csv_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([eye_direction, emotion, gesture])
