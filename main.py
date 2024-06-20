import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD, accuracy
from surprise.model_selection import train_test_split

users = pd.DataFrame({
    'UserID': [1, 2, 3, 4, 5],
    'FitnessLevel': ['Beginner', 'Intermediate', 'Advanced', 'Beginner', 'Intermediate'],
    'Goals': ['Weight Loss', 'Muscle Gain', 'Endurance', 'Flexibility', 'Weight Loss'],
    'AvailableEquipment': ['Dumbbells', 'Barbell', 'None', 'Resistance Bands', 'Kettlebells'],
    'TimeConstraints': [30, 60, 45, 20, 50]
})

workouts = pd.DataFrame({
    'WorkoutID': [101, 102, 103, 104, 105],
    'WorkoutType': ['Cardio', 'Strength', 'Yoga', 'HIIT', 'Pilates'],
    'Duration': [30, 60, 45, 20, 50],
    'Intensity': ['High', 'Medium', 'Low', 'High', 'Medium'],
    'RequiredEquipment': ['None', 'Dumbbells', 'Mat', 'None', 'Resistance Bands']
})

ratings = pd.DataFrame({
    'UserID': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'WorkoutID': [101, 102, 103, 104, 105, 102, 103, 104, 105, 101],
    'Rating': [4, 5, 3, 2, 4, 5, 4, 3, 2, 1]
})

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'WorkoutID', 'Rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25)

algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)
accuracy.rmse(predictions)


def get_top_n_recommendations(user_id, n=5):
    workout_ids = workouts['WorkoutID'].unique()
    user_workouts = [(user_id, workout_id, algo.predict(user_id, workout_id).est) for workout_id in workout_ids]
    user_workouts.sort(key=lambda x: x[2], reverse=True)
    top_n_workouts = user_workouts[:n]
    return top_n_workouts


recommendations = get_top_n_recommendations(user_id=1, n=5)
print(recommendations)

