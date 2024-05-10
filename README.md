# Food-Reccomdation-System-Using-DMF
import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, f1_score

# Step 1: Load datasets
food_df = pd.read_csv('food.csv')
ratings_df = pd.read_csv('ratings.csv')

# Step 2: Preprocess data for DMF
num_users = ratings_df['User_ID'].nunique()
num_items = ratings_df['Food_ID'].nunique()

user_input = Input(shape=[1], name='User')
item_input = Input(shape=[1], name='Item')

user_embedding = Embedding(output_dim=50, input_dim=num_users + 1,
                           input_length=1, name='UserEmbedding')(user_input)
item_embedding = Embedding(output_dim=50, input_dim=num_items + 1,
                           input_length=1, name='ItemEmbedding')(item_input)

user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

# Step 3: Define DMF architecture
concat = Concatenate()([user_vecs, item_vecs])
hidden = Dense(50, activation='relu')(concat)
output = Dense(1)(hidden)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam')

# Step 4: Train DMF model
ratings_df['User_ID'] = ratings_df['User_ID'].astype('category').cat.codes.values
ratings_df['Food_ID'] = ratings_df['Food_ID'].astype('category').cat.codes.values
history = model.fit([ratings_df['User_ID'], ratings_df['Food_ID']], ratings_df['Rating'],
                    batch_size=64, epochs=5, validation_split=0.2)

# Step 5: Get user input for rating


# Step 6: Get user input for food preference (veg/nonveg)
while True:
    user_preference = input("Enter your food preference (veg/non-veg): ").lower()
    if user_preference not in ['veg', 'non-veg']:
        print("Invalid preference. Please enter 'veg' or 'non-veg'.")
        continue
    break

# Step 7: Get user input for cuisine preference
user_cuisine = input("Enter your preferred cuisine: ")

while True:
    try:
        user_rating = int(input("Enter your desired rating (1-10): "))
        if user_rating < 1 or user_rating > 10:
            print("Rating must be between 1 and 10.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a valid rating.")
positive_food_ids = ratings_df[ratings_df['Rating'] >= user_rating]['Food_ID'].unique()
food_ids_with_rating = ratings_df[ratings_df['Rating'] == user_rating]['Food_ID'].unique()
# Step 8: Get recommended food IDs using DMF model
food_ids_with_rating = ratings_df[ratings_df['Rating'] == user_rating]['Food_ID'].unique()
user_id = np.array([0])  # Assume user's ID is 0
item_ids = food_df[(food_df['Food_ID'].isin(food_ids_with_rating)) & 
                   (food_df['Veg_Non'] == user_preference) & 
                   (food_df['C_Type'] == user_cuisine)]['Food_ID'].values

# Step 9: Predict ratings for selected food IDs
predicted_ratings = model.predict([np.repeat(user_id, len(item_ids)), item_ids])

# Step 10: Sort recommended food IDs based on predicted ratings
sorted_indices = np.argsort(predicted_ratings.flatten())[::-1]
recommended_food_ids = item_ids[sorted_indices]

# Step 11: Print names of recommended foods
recommended_foods = food_df[food_df['Food_ID'].isin(recommended_food_ids)]
if recommended_foods.empty:
    print("Sorry, no matching foods found for your specified rating, preference, and cuisine.")
else:
    print("Recommended foods with rating {}, preference: {}, and cuisine: {}:".format(user_rating, user_preference, user_cuisine))
    for index, row in recommended_foods.iterrows():
        print("Food Name:", row['Name'])
        print("Description:", row['Describe'])
        print("\n")

# Step 12: Compute evaluation metrics
# Load the test data (assuming you have a separate test set)
from sklearn.model_selection import train_test_split

# Split the ratings data into training and testing sets
train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Save the test set to a CSV file
test_ratings.to_csv('test_ratings.csv', index=False)

test_ratings = pd.read_csv('test_ratings.csv')

# Convert user and item IDs to categorical codes
test_ratings['User_ID'] = test_ratings['User_ID'].astype('category').cat.codes.values
test_ratings['Food_ID'] = test_ratings['Food_ID'].astype('category').cat.codes.values

# Predict ratings for the test set
test_predictions = model.predict([test_ratings['User_ID'], test_ratings['Food_ID']]).flatten()

# Compute RMSE
rmse = np.sqrt(mean_squared_error(test_ratings['Rating'], test_predictions))

# Compute MAE
mae = mean_absolute_error(test_ratings['Rating'], test_predictions)

# Convert predictions to binary (0/1) for F1-score and precision
test_predictions_binary = np.where(test_predictions >= 0.5, 1, 0)
test_ratings_binary = np.where(test_ratings['Rating'] >= 0.5, 1, 0)

# Compute F1-score
f1 = f1_score(test_ratings_binary, test_predictions_binary)

# Compute precision
#precision = precision_score(test_ratings_binary, test_predictions_binary)


# Convert RMSE to percentage
rmse_percentage = (rmse / (ratings_df['Rating'].max() - ratings_df['Rating'].min())) * 100

# Convert MAE to percentage
mae_percentage = (mae / ratings_df['Rating'].mean()) * 100

# Convert F1-score to percentage
f1_percentage = f1 * 100
# Assuming you have computed precision earlier
#precision_percentage = precision * 100
from sklearn.metrics import recall_score

# Assuming you have computed predictions and ground truth labels earlier
recall = recall_score(test_ratings_binary, test_predictions_binary)

# Convert recall to percentage
recall_percentage = recall * 100

# Print the recall percentage
print("Recall Percentage:", recall_percentage)

# Print the precision percentage
#print("Precision Percentage:", precision_percentage)
# Print the converted values
#print("RMSE Percentage:", rmse_percentage)
#print("MAE Percentage:", mae_percentage)
print("F1-score Percentage:", f1_percentage)


from sklearn.metrics import confusion_matrix
# Assuming you have computed predictions and ground truth labels earlier
conf_matrix = confusion_matrix(test_ratings_binary, test_predictions_binary)
# Example confusion matrix
def precision_at_top_n(recommended_ids, relevant_ids, n):
    recommended_ids = recommended_ids[:n]
    num_relevant_and_recommended = len(set(recommended_ids) & set(relevant_ids))
    precision = num_relevant_and_recommended / n if n > 0 else 0
    return precision

# Define relevant items and recommended items
relevant_ids = positive_food_ids
recommended_ids = recommended_food_ids

# Evaluate precision at different values of N
#top_n_values = [7, 10, 12]  # You can adjust this list as needed
#for n in top_n_values:
#    precision_n = precision_at_top_n(recommended_ids, relevant_ids, n)
#    print("Precision at Top-{}: {:.2f}".format(n, (precision_n*100)))




