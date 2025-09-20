import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("winequality-red.csv", sep=";")

print("shape of the data:" ,df.shape)
print("Types for data: ", df.dtypes)

print("description: ", df.describe())
print("Missing values: ", df.isnull().sum())

print("Quality Distribution: ", df['quality'].value_counts().sort_values())

plt.figure(figsize=(8,6))
sn.histplot(df['quality'], kde=True)
plt.title("Histogram of Quality")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()


# # Correlation heat map

# # plt.figure(figsize=(12, 10))
# # sn.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# # plt.title("Feature Correlation")
# # plt.show()


# #box plot of alcohol vs quality

# # plt.figure(figsize=(8, 6))
# # sn.boxplot(x='quality', y='alcohol', data=df)
# # plt.title('Alcohol vs. Quality')
# # plt.xlabel('Quality Score')
# # plt.ylabel('Alcohol Content')
# # plt.show()


# X = df.drop('quality', axis=1)
# y = df['quality']

# X.fillna(X.mean())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_trained_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)



# print("X_train Shape:", X_trained_scaled.shape)
# print("X_test Shape:", X_test_scaled.shape)

# lr_model = LinearRegression()

# lr_model.fit(X_trained_scaled, y_train)

# y_pred_lr = lr_model.predict(X_test_scaled)

# mse_lr = mean_squared_error(y_test, y_pred_lr)

# r2_lr = r2_score(y_test, y_pred_lr)

# print("Linear Regression - MSE:" , mse_lr)
# print("Linear Regression - R:", r2_lr)

# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# rf_model.fit(X_trained_scaled, y_train)

# y_pred_rf = rf_model.predict(X_test_scaled)

# mse_rf = mean_squared_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)

# print("Random Forest - MSE:", mse_rf)
# print("Random Forest - R²:", r2_rf)


# importances = pd.Series(rf_model.feature_importances_, index=X_trained_scaled.columns)
# print("Feature Importances:\n", importances.sort_values(ascending=False))