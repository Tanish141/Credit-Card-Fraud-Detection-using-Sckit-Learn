# Imoprt Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset 
df = pd.read_csv('Dataset/creditcard.csv')

# Display the information of dataset
print(df.info())
print("\n",df.describe())

# Display the first few rows of the dataset
print("\nDataset Sample:\n",df.head())

# Seprate features (X) and target (y)
X = df.drop('Class',axis=1)
y = df['Class']

# Split the dataset into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

# Scale the features for numerical stability
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
conf_mat = confusion_matrix(y_test,y_pred)

print(f"\nAccuracy score : {accuracy}")
print(f"\nClassification Report : \n",report)
print(f"\nConfusion matrix : \n",conf_mat)


