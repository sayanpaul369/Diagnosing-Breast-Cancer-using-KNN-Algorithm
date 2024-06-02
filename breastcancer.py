import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read the CSV file into a DataFrame
df = pd.read_csv("cancer.csv")

# Display the column names to identify the correct column name for diagnosis
print(df.columns)

# Remove the parentheses and replace with '_'
df = df.rename(columns={'diagnosis(1=m, 0=b)': 'diagnosis'})

# Check the first few rows of the DataFrame
print(df.head(10))

# Check the shape of the DataFrame
print(df.shape)

# Check for missing values
print(df.isna().sum())

# Drop columns with missing values
df = df.dropna(axis=1)
print(df.shape)

# Check value counts for diagnosis
print(df['diagnosis'].value_counts())

# Encode the categorical values in 'diagnosis' column
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Check data types
print(df.dtypes)

# Visualize data using pairplot
sns.pairplot(df.iloc[:, 0:6], hue='diagnosis')

# Prepare data for modeling
x = df.iloc[:, 2:].values  # Exclude 'id' and 'diagnosis' columns from features
y = df.iloc[:, 1].values    # Target column is 'diagnosis'

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Train the KNN classifier
knn = neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)

# Calculate and print the accuracy score
accuracy = knn.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions on test data
y_pred = knn.predict(x_test)

# Calculate and print the confusion matrix and classification report
con_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
print("Confusion Matrix:\n", con_matrix)
print("Classification Report:\n", classification_report_str)

# Prompt user to input new data features
new_data_features = {}
feature_names = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                 "smoothness_mean", "compactness_mean", "concavity_mean", 
                 "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 
                 "radius_se", "texture_se", "perimeter_se", "area_se",
                 "smoothness_se", "compactness_se", "concavity_se", 
                 "concave points_se", "symmetry_se", "fractal_dimension_se", 
                 "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                 "smoothness_worst", "compactness_worst", "concavity_worst", 
                 "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]

# Get user input for each feature
for feature_name in feature_names:
    user_input = float(input(f"Enter the value for {feature_name}: "))
    new_data_features[feature_name] = user_input

# Create a DataFrame with the new data features
new_data = pd.DataFrame([new_data_features])

# Standardize the new data features
new_data_scaled = sc.transform(new_data)

# Make prediction on new data
prediction = knn.predict(new_data_scaled)

# Print the prediction result
if prediction[0] == 1:
    print("\n\n\n\nThe person is predicted to have breast cancer (Malignant).")
else:
    print("\n\n\n\nThe person is predicted to be cancer-free (Benign).")
