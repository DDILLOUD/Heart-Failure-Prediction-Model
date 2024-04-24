from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the scaled training data
rf_model.fit(X_train_scaled, y_train)

# Predict on the scaled testing data
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


from sklearn.model_selection import GridSearchCV

# Define the hyperparameters grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt']  # Set max_features explicitly to 'sqrt'
}

# Initialize the Random Forest classifier
rf_model_tuned = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model_tuned, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform grid search to find the best hyperparameters
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Predict on the scaled testing data using the best model
y_pred_best = best_rf_model.predict(X_test_scaled)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Random Forest Classifier Accuracy:", accuracy_best)

# Classification report for the best model
print("\nClassification Report for the Best Model:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix for the best model
print("\nConfusion Matrix for the Best Model:")
print(confusion_matrix(y_test, y_pred_best))


#This code segment is used to plot the Precision-Recall curve and calculate the
Average Precision Score for evaluating the performance of a machine learning model,
specifically a binary classifier like the Heart Failure Prediction Model.

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Predict probabilities for the positive class
y_prob_best = best_rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, y_prob_best)

# Plot Precision-Recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Calculate average precision score
average_precision = average_precision_score(y_test, y_prob_best)
print(f'Average Precision Score: {average_precision:.2f}')


