# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

# Matplotlib for visualization
import matplotlib.pyplot as plt

# Set default font size
plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Imputing missing values and scaling values
from sklearn.preprocessing import MinMaxScaler

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Read in data into dataframes 
train_features = pd.read_csv('data/training_features.csv')
test_features = pd.read_csv('data/testing_features.csv')
train_labels = pd.read_csv('data/training_labels.csv')
test_labels = pd.read_csv('data/testing_labels.csv')

# Display sizes of data
print('Training Feature Size: ', train_features.shape)
print('Testing Feature Size:  ', test_features.shape)
print('Training Labels Size:  ', train_labels.shape)
print('Testing Labels Size:   ', test_labels.shape)

train_features.head(12)



# Create an imputer object with a median filling strategy
imputer = SimpleImputer(strategy='median')

# Train on the training features
imputer.fit(train_features)

# Transform both training data and testing data
X = imputer.transform(train_features)
X_test = imputer.transform(test_features)
print('Missing values in training features: ', np.sum(np.isnan(X)))
print('Missing values in testing features:  ', np.sum(np.isnan(X_test)))

# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)
# Convert y to one-dimensional array (vector)
y = np.array(train_labels).reshape((-1, ))
y_test = np.array(test_labels).reshape((-1, ))



# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X, y)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
    return model_mae
# lr = LinearRegression()
# lr_mae = fit_and_evaluate(lr)

# print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)

# svm = SVR(C = 1000, gamma = 0.1)
# svm_mae = fit_and_evaluate(svm)

# print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)

# random_forest = RandomForestRegressor(random_state=60)
# random_forest_mae = fit_and_evaluate(random_forest)

# print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)

# gradient_boosted = GradientBoostingRegressor(random_state=60)
# gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

# print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)

# knn = KNeighborsRegressor(n_neighbors=10)
# knn_mae = fit_and_evaluate(knn)

# print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)

# plt.style.use('fivethirtyeight')
# figsize(8, 6)

# # Dataframe to hold the results
# model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine',
#                                            'Random Forest', 'Gradient Boosted',
#                                             'K-Nearest Neighbors'],
#                                  'mae': [lr_mae, svm_mae, random_forest_mae, 
#                                          gradient_boosted_mae, knn_mae]})

# # Horizontal bar chart of test mae
# model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',
#                                                            color = 'red', edgecolor = 'black')

# # Plot formatting
# plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Mean Absolute Error'); plt.xticks(size = 14)
# plt.title('Model Comparison on Test MAE', size = 20);
# plt.show()









# Loss function to be optimized
loss = ['ls', 'lad', 'huber']

# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Определение сетки гиперпараметров для поиска
hyperparameter_grid = {
    'loss': ['squared_error', 'absolute_error', 'huber'],
    # Другие параметры, которые вы хотите настроить
    'max_features': ['sqrt', 'log2', None],
    # Пример других параметров
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0],
    'max_depth': [3, 4, 5]
}

# Создание модели для настройки гиперпараметров
model = GradientBoostingRegressor(random_state=42)

# Настройка случайного поиска с 4-кратной кросс-валидацией
random_cv = RandomizedSearchCV(
    estimator=model,
    param_distributions=hyperparameter_grid,
    cv=4,
    n_iter=15,  # Количество итераций поиска
    scoring='neg_mean_squared_error',  # Метрика для оценки
    random_state=42
)

# Подготовка данных для обучения
# X - признаки, y - целевая переменная
# Не забудьте предварительно обработать данные, если это необходимо

# Запуск поиска
random_cv.fit(X, y)

random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)

random_results.head(10)
random_cv.best_estimator_


# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}

model = GradientBoostingRegressor(loss = 'lad', max_depth = 5,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = None,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
# Fit the grid search
grid_search.fit(X, y)


# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend();
plt.title('Performance vs Number of Trees');