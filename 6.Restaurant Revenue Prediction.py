import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Load dataset
dataset = pd.read_csv("cleaned_restaurant_data.csv")

# One-hot encoding (if categorical columns exist)
df = pd.get_dummies(dataset, dtype=int, drop_first=True)

# Separate independent and dependent variables
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Select top 5 features using f_regression
kbest = SelectKBest(score_func=f_regression, k=5)
X_kbest = kbest.fit_transform(X, y)
selected_features = X.columns[kbest.get_support()]
X_kbest = pd.DataFrame(X_kbest, columns=selected_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.25, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

# Initialize lists to store R2 scores
acclin = []
accsvml = []
accsvmnl = []
accdes = []
accrf = []

# Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
acclin.append(r2_score(y_test, y_pred))

# Support Vector Machine (linear) model
svm_l = SVR(kernel='linear')
svm_l.fit(X_train, y_train)
y_pred = svm_l.predict(X_test)
accsvml.append(r2_score(y_test, y_pred))

# Support Vector Machine (non-linear) model
svm_nl = SVR(kernel='rbf')
svm_nl.fit(X_train, y_train)
y_pred = svm_nl.predict(X_test)
accsvmnl.append(r2_score(y_test, y_pred))

# Decision Tree model
d_tree = DecisionTreeRegressor(random_state=0)
d_tree.fit(X_train, y_train)
y_pred = d_tree.predict(X_test)
accdes.append(r2_score(y_test, y_pred))

# Random Forest model
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accrf.append(r2_score(y_test, y_pred))

# Combine all the results into a DataFrame for easy comparison
result = pd.DataFrame(index=['R2 Score'], columns=['Linear', 'SVMl', 'SVMnl', 'Decision', 'Random'])
result['Linear'] = acclin
result['SVMl'] = accsvml
result['SVMnl'] = accsvmnl
result['Decision'] = accdes
result['Random'] = accrf

# Print the results
print(result)

# Save Random Forest model and scaler
with open('rf_regression_model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load the saved model and scaler
with open('rf_regression_model.pkl', 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

new_input = np.array([[38, 73.98, 1, 1, 0]])  

# Scale the new input data
scaled_input = loaded_scaler.transform(new_input)

# Make the prediction using the trained model
prediction = loaded_rf_model.predict(scaled_input)

# Print the prediction
print("Prediction for the new input:", prediction)