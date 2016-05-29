from sklearn.datasets import load_boston
from sklearn.datasets.samples_generator import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import cProfile


def generate_random_dataset(n_train, n_test, n_features, noise=0.1,
                            verbose=False):
    """Generate a regression dataset with the given parameters."""
    if verbose:
        print("generating dataset...")
    X, y, coef = make_regression(n_samples=n_train + n_test,
                                 n_features=n_features, noise=noise,
                                 coef=True, random_state=0)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    idx = np.arange(n_train)
    np.random.seed(0)
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]

    std = X_train.std(axis=0)
    mean = X_train.mean(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    std = y_train.std(axis=0)
    mean = y_train.mean(axis=0)
    y_train = (y_train - mean) / std
    y_test = (y_test - mean) / std

    return X_train, y_train, X_test, y_test

# comment out one of these blocks depending on the data you want to use
# use boston dataset
boston = load_boston()
X_train = boston.data
X_test = boston.data
X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    test_size=0.25,
                                                    random_state=0)

# create a random, larger dataset so the time differences are more apparent
# n_train = int(1e3)
# n_test = int(1e2)
# n_features = int(1e2)
# X_train, y_train, X_test, y_test = generate_random_dataset(n_train,
#                                                            n_test,
#                                                            n_features)

# Fit regression model
mse_regressor = DecisionTreeRegressor(random_state=0)
mae_regressor = DecisionTreeRegressor(random_state=0, criterion="mae")

cProfile.run('mse_regressor.fit(X_train, y_train)')
cProfile.run('mae_regressor.fit(X_train, y_train)')

mse_predicted = mse_regressor.predict(X_test)
mae_predicted = mae_regressor.predict(X_test)

mse_mse = mean_squared_error(y_test, mse_predicted)
mae_mse = mean_squared_error(y_test, mae_predicted)
print "Mean Squared Error of Tree Trained w/ MSE Criterion: {}".format(mse_mse)
print "Mean Squared Error of Tree Trained w/ MAE Criterion: {}".format(mae_mse)

mse_mae = mean_absolute_error(y_test, mse_predicted)
mae_mae = mean_absolute_error(y_test, mae_predicted)
print "Mean Absolute Error of Tree Trained w/ MSE Criterion: {}".format(mse_mae)
print "Mean Absolute Error of Tree Trained w/ MAE Criterion: {}".format(mae_mae)
