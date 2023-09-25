# Coded by: Qyfashae

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
data_columns = ["Fwd Seg Size Min", "Init Bwd Win Byts", "Init Fwd Win Byts", "Fwd Seg Size Min", "Fwd Seg Size Avg", "Label", "Timestamp"]
data_dtypes = {"Fwd Pkt Len Mean": float, "Fwd Seg Size Avg": float, "Init Fwd Win Byts": int, "Init Bwd Win Byts": int, "Fwd Seg Size Min": int, "Label": str}
date_col = ["Timestamp"]

raw_data = pd.read_csv("att_data.csv", usecols=data_columns, dtype=data_dtypes, parse_dates=date_col, index_col=None)
sorted_data = raw_data.sort_values("Timestamp")
processed_data = sorted_data.drop(columns=["Timestamp"])

# Split the data into training and testing sets
X = processed_data.drop(columns=["Label"])
y = processed_data["Label"]

'''
If you want to use the ".head, .tail" here use 1 * 0.8 to a value of 1 * 0.2 = for 80/20 testing with a collection of 1 * float to a value of 1 * float. 
Could also use flat div but either set the test_size and random state below or use the ".head, .tail" (Its whatever but you can try both).
Will look like --> 
train_df = processed_data.head(1 * 0.8)
test_df = processed_data.tail(1 * 0.2)
But you will need to define the lenght of data. You can use --> Above the df data above -->
l_data = len(processed_data.index)
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Instantiate and train the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=45)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)
