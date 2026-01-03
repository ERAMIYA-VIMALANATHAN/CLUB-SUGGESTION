from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import os

def get_interests(row):
    academic = row['AcademicInterest']
    skills = str(row['Skills']).lower()
    extracurricular = str(row['ExtracurricularActivities']).lower()

    base = {
        "Computer Science": [1, 1, 5, 1],
        "Mathematics": [1, 3, 4, 2],
        "Physics": [1, 2, 5, 1],
        "Biology": [2, 2, 3, 4],
        "Psychology": [3, 2, 2, 5],
        "History": [4, 3, 1, 4]
    }.get(academic, [2, 2, 2, 2])

    music, sports, tech, art = base

    if 'programming' in skills or 'coding' in skills:
        tech = min(5, tech + 2)
    if 'data analysis' in skills or 'statistics' in skills:
        tech = min(5, tech + 1)
    if 'artistic' in skills or 'creative' in skills:
        art = min(5, art + 2)
    if 'leadership' in skills:
        sports = min(5, sports + 1)
    if 'public speaking' in skills or 'communication' in skills:
        music = min(5, music + 1)

    if 'music' in extracurricular or 'band' in extracurricular:
        music = min(5, music + 2)
    if 'sports' in extracurricular or 'athletics' in extracurricular:
        sports = min(5, sports + 2)
    if 'coding' in extracurricular or 'tech' in extracurricular:
        tech = min(5, tech + 2)
    if 'art' in extracurricular or 'design' in extracurricular:
        art = min(5, art + 2)

    return [music, sports, tech, art]

csv_file = "student_data_extended.csv"

if not os.path.exists(csv_file):
    print(f"Error: '{csv_file}' not found.")
    exit(1)

df = pd.read_csv(csv_file)

data = []
for _, row in df.iterrows():
    interests = get_interests(row)
    clubs = str(row['ClubMemberships']).split(',')
    if clubs and clubs[0].strip().lower() != 'nan':
        data.append((interests, clubs[0].strip()))

if not data:
    print("Error: No valid data found in CSV.")
    exit(1)

print("\nStudent Interest Quiz")
print("Rate your interest from 0 to 5")

try:
    music = int(input("1. Music interest rating (0–5): "))
    sports = int(input("2. Sports interest rating (0–5): "))
    tech = int(input("3. Technology/Coding interest rating (0–5): "))
    art = int(input("4. Art/Design interest rating (0–5): "))

    if not all(0 <= x <= 5 for x in [music, sports, tech, art]):
        raise ValueError
except:
    print("Invalid input. Ratings must be between 0 and 5.")
    exit(1)

user_interests = np.array([music, sports, tech, art]).reshape(1, -1)

distances = []
for interests, club in data:
    dist = euclidean_distances(
        user_interests,
        np.array(interests).reshape(1, -1)
    )[0][0]
    distances.append((dist, club))

_, best_club = min(distances, key=lambda x: x[0])

print("\nBased on similarity to students in the dataset, you may enjoy:")
print(best_club)
print("\nThank you for using the Student Interest Recommender")
