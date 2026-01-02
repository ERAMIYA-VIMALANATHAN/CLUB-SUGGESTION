
from sklearn.metrics.pairwise import euclidean_distances  # Added for similarity calculation
import numpy as np
import pandas as pd

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

try:
    df = pd.read_csv('ML_classification/student_data_extended.csv')  # Fixed path to include subdirectory since script is run from parent directory
except FileNotFoundError:
    print("Error: 'ML_classification/student_data_extended.csv' not found. Please ensure the file exists in the ML_classification subdirectory.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

data = []
for index, row in df.iterrows():
    interests = get_interests(row)
    clubs = str(row['ClubMemberships']).split(',')
    if clubs and clubs[0].strip() and clubs[0].strip().lower() != 'nan':
        club = clubs[0].strip()
        data.append((interests, club))

if not data:
    print("Error: No valid data found in CSV for recommendations.")
    exit(1)

print("\n--- Student Interest Quiz ---")
print("Rate your interest from 0 (low) to 5 (high)")

try:
    music = int(input("1. Music interest rating (0–5): "))
    if not (0 <= music <= 5):
        raise ValueError("Rating must be between 0 and 5.")
    sports = int(input("2. Sports interest rating (0–5): "))
    if not (0 <= sports <= 5):
        raise ValueError("Rating must be between 0 and 5.")
    tech = int(input("3. Technology/Coding interest rating (0–5): "))
    if not (0 <= tech <= 5):
        raise ValueError("Rating must be between 0 and 5.")
    art = int(input("4. Art/Design interest rating (0–5): "))
    if not (0 <= art <= 5):
        raise ValueError("Rating must be between 0 and 5.")
except ValueError as e:
    print(f"Invalid input: {e}")
    exit(1)

user_interests = np.array([music, sports, tech, art]).reshape(1, -1)

distances = []
for interests, club in data:
    dist = euclidean_distances(user_interests, np.array(interests).reshape(1, -1))[0][0]
    distances.append((dist, club))

min_dist, best_club = min(distances, key=lambda x: x[0])

print("\n Based on similarity to students in the dataset, you may enjoy:")
print("=>", best_club)
print("\nThank you for using the Student Interest Recommender!")