import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
## PART 0: Data Loading and Preparation
def load_openml_dataset():

    dataset = fetch_openml(name='vehicle', version=1, as_frame=True, parser='auto')
    X = dataset.data.values
    
    target_values = dataset.target.values
    unique_targets = np.unique(target_values)
    
    # Create mapping from string labels to integers
    label_to_int = {label: i for i, label in enumerate(unique_targets)}
    y = np.array([label_to_int[label] for label in target_values])
    
    feature_names = dataset.feature_names
    
    return X, y, feature_names

X, y, feature_names = load_openml_dataset()
print(f"Feature shape: {X.shape}, target output shape:{y.shape}")
print(f"feature names: {feature_names}")

## PART 1: Dataset Partition and One-hot Encoding
def dataset_partition_encoding(X, y):
    """
    Input type
    :X type: numpy.ndarray of size (number_of_samples, number_of_features)
    :y type: numpy.ndarray of size (number_of_samples,)

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, number_of_features)
    :X_val type: numpy.ndarray of size (number_of_validation_samples, number_of_features)
    :X_test type: numpy.ndarray of size (number_of_test_samples, number_of_features)
    :Ytr_onehot type: numpy.ndarray of size (number_of_training_samples, num_classes)
    :Yval_onehot type: numpy.ndarray of size (number_of_validation_samples, num_classes)
    :Yts_onehot type: numpy.ndarray of size (number_of_test_samples, num_classes)

    """

    # your code goes here
    seed_matrix_no = 924  # replace with last 3 matric digits

    # Split 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed_matrix_no)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed_matrix_no)

    # One-hot encode target labels
    encoder = OneHotEncoder(sparse_output=False)
    Ytr_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    Yval_onehot = encoder.transform(y_val.reshape(-1, 1))
    Yts_onehot = encoder.transform(y_test.reshape(-1, 1))

    # return in this order
    return X_train, X_val, X_test, Ytr_onehot, Yval_onehot, Yts_onehot

X_train, X_val, X_test, Ytr_onehot, Yval_onehot, Yts_onehot = dataset_partition_encoding(X, y)
print(f"Training set shape: {X_train.shape}, {Ytr_onehot.shape}")
print(f"Validation set shape: {X_val.shape}, {Yval_onehot.shape}")
print(f"Test set shape: {X_test.shape}, {Yts_onehot.shape}")
## PART 2: Feature Selection using Pearson Correlation
def feature_selection(X_train, X_val, X_test, feature_names, threshold=0.8):
    """
    Input type
    :X_train type: numpy.ndarray of size (number_of_training_samples, number_of_features)
    :X_val type: numpy.ndarray of size (number_of_validation_samples, number_of_features)
    :X_test type: numpy.ndarray of size (number_of_test_samples, number_of_features)
    :feature_names type: list of str
    :threshold type: float

    Return type
    :selected_features type: list of str
    :FS_X_train type: numpy.ndarray of size (number_of_training_samples, number_of_selected_features)
    :FS_X_val type: numpy.ndarray of size (number_of_validation_samples, number_of_selected_features)
    :FS_X_test type: numpy.ndarray of size (number_of_test_samples, number_of_selected_features)

    """
    #q7
    # your code goes here
    
    df_train = pd.DataFrame(X_train, columns=feature_names)
    corr_matrix = df_train.corr()
    selected_idx = [0]
    # Iterate through remaining features (1–17)
    for j in range(1, len(feature_names)):
        # Correlations of feature j with all selected features
        selected_corr = corr_matrix.iloc[j, selected_idx].abs()
        if selected_corr.max() <= threshold:
            selected_idx.append(j)

    selected_features = [feature_names[i] for i in selected_idx]

    # apply selection to all splits
    FS_X_train = X_train[:, selected_idx]
    FS_X_val   = X_val[:,   selected_idx]
    FS_X_test  = X_test[:,  selected_idx]

    # return in this order
    return selected_features, FS_X_train, FS_X_val, FS_X_test

selected_features, FS_X_train, FS_X_val, FS_X_test = feature_selection(X_train, X_val, X_test, feature_names)

print(f"{len(selected_features)} Selected Features: {selected_features}")
print(f"Training set shape after feature selection: {FS_X_train.shape}, {Ytr_onehot.shape}")
print(f"Validation set shape after feature selection: {FS_X_val.shape}, {Yval_onehot.shape}")
print(f"Test set shape after feature selection: {FS_X_test.shape}, {Yts_onehot.shape}")

## PART 3: Polynomial Feature Transformation and Classification
def polynomial_for_classification(FS_X_train, FS_X_val, FS_X_test, Ytr_onehot, Yval_onehot, Yts_onehot, max_order=3, lamda=0.001):
    """
    Args:
        FS_X_train (np.ndarray): Feature matrix for training.
        FS_X_val (np.ndarray): Feature matrix for validation.
        FS_X_test (np.ndarray): Feature matrix for testing.
        Ytr_onehot (np.ndarray): One-hot encoded labels for training.
        Yval_onehot (np.ndarray): One-hot encoded labels for validation.
        Yts_onehot (np.ndarray): One-hot encoded labels for testing.
        max_order (int): Maximum polynomial order to consider.
        lamda (float): Regularization strength.

    Returns:
        acc_train_list (list): Training accuracies for each polynomial order.
        acc_val_list (list): Validation accuracies for each polynomial order.
        best_order (int): Best polynomial order based on validation accuracy.
        acc_test (float): Test accuracy for the best polynomial order.

    """

    # your code goes here
    # Q6 part3

    # true labels from one-hot
    y_tr  = np.argmax(Ytr_onehot,  axis=1)
    y_val = np.argmax(Yval_onehot, axis=1)
    y_ts  = np.argmax(Yts_onehot,  axis=1)

    acc_train_list = []
    acc_val_list   = []
    best_order = None
    best_val = -1.0
    best_test_pred = None

    for order in range(1, max_order + 1):
        # Polynomial features (include bias to match Q6 augment step)
        Poly = PolynomialFeatures(degree=order, include_bias=True)
        P_tr  = Poly.fit_transform(FS_X_train)
        P_val = Poly.transform(FS_X_val)
        P_ts  = Poly.transform(FS_X_test)

        PtP = P_tr.T @ P_tr
        PtY = P_tr.T @ Ytr_onehot  # multi-output: columns = classes

        # Solve: try unregularised first; if singular, use ridge with λ
        try:
            W = np.linalg.solve(PtP, PtY)
        except np.linalg.LinAlgError:
            W = np.linalg.solve(PtP + lamda * np.eye(PtP.shape[0]), PtY)

        # Scores & predictions
        yhat_tr  = np.argmax(P_tr  @ W, axis=1)
        yhat_val = np.argmax(P_val @ W, axis=1)
        yhat_ts  = np.argmax(P_ts  @ W, axis=1)

        acc_tr  = accuracy_score(y_tr,  yhat_tr)
        acc_val = accuracy_score(y_val, yhat_val)

        acc_train_list.append(acc_tr)
        acc_val_list.append(acc_val)

        # pick best by validation accuracy (tie-breaker: lower order)
        if acc_val > best_val:
            best_val = acc_val
            best_order = order
            best_test_pred = yhat_ts

    acc_test = accuracy_score(y_ts, best_test_pred)


    # return in this order              
    return acc_train_list, acc_val_list, best_order, acc_test

acc_train_list, acc_val_list, best_order, acc_test = polynomial_for_classification(FS_X_train, FS_X_val, FS_X_test, Ytr_onehot, Yval_onehot, Yts_onehot)

print(f"Training accuracies: {np.round(acc_train_list,2)}")
print(f"Validation accuracies: {np.round(acc_val_list,2)}")
print(f"Best polynomial order: {best_order}")
print(f"Test accuracy for best order {best_order}: {np.round(acc_test,2)}")

## PART 4: Multinomial Logistic Regression
def MLR_select_lr(FS_X_train, FS_X_val, FS_X_test, Ytr_onehot, Yval_onehot, Yts_onehot, lr_list=[0.0001, 0.001, 0.01, 0.1], num_iters=20000):
    """
    Args:
        FS_X_train (np.ndarray): Feature matrix for training.
        FS_X_val (np.ndarray): Feature matrix for validation.
        FS_X_test (np.ndarray): Feature matrix for testing.
        Ytr_onehot (np.ndarray): One-hot encoded labels for training.
        Yval_onehot (np.ndarray): One-hot encoded labels for validation.
        Yts_onehot (np.ndarray): One-hot encoded labels for testing.
        lr_list (list): List of learning rates to test.
        num_iters (int): Number of iterations for training.

    Returns:
        cost_dict (dict): Dictionary of cost values for each learning rate without input normalization.
                          example: cost_dict = {0.0001: [0.1, 0.05, ...], 0.001: [0.09, 0.045, ...], ...}
        acc_train_list_Log (list): Training accuracies for each learning rate without input normalization.
        acc_val_list_Log (list): Validation accuracies for each learning rate without input normalization.
        best_lr (float): Best learning rate based on validation accuracy without input normalization.
        test_acc_Log (float): Test accuracy for the best learning rate without input normalization.
        cost_dict_norm (dict): Dictionary of cost values for each learning rate with input normalization.
        acc_train_list_Log_norm (list): Training accuracies for each learning rate with input normalization.
        acc_val_list_Log_norm (list): Validation accuracies for each learning rate with input normalization.
        best_lr_norm (float): Best learning rate based on validation accuracy with input normalization.
        test_acc_Log_norm (float): Test accuracy for the best learning rate with input normalization.

    """
    
    # your code goes here
    # Q9
    #---------------------------------------Helper Func & Initialisation------------------------------------------

    # Compute prediction, cost and gradient based on categorical cross entropy
    def multi_logistic_cost_gradient(X, W, Y, eps=1e-15):
        z = X @ W
        z_max = np.max(z, axis=-1, keepdims=True)  # for numerical stability
        exp_z = np.exp(z - z_max)
        pred_Y = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

        # Clip predictions to prevent log(0)
        pred_Y = np.clip(pred_Y, eps, 1 - eps)
    
        N = X.shape[0]  # Number of samples
        cost   = (np.sum(-(Y * np.log(pred_Y))))/N
        gradient = (X.T @ (pred_Y-Y))/N

        return pred_Y, cost, gradient   
    
    def multinomial_logistic_regression(P, W, Y, lr, num_iters):
        pred_Y, cost, gradient = multi_logistic_cost_gradient(P, W, Y)
        #print('Initial Cost =', cost)
        #print('Initial Weights =', W)
        cost_vec = np.zeros(num_iters+1)
        cost_vec[0] = cost

        for i in range(1, num_iters + 1):
            W -= lr * gradient
            pred_Y, cost, gradient = multi_logistic_cost_gradient(P, W, Y)
            cost_vec[i] = cost
            #if i % 2000 == 0:
            #print(f"Iteration {i}, Cost: {cost}")

        return W, cost_vec, pred_Y  
    
    #Initialisation
    seed_matrix_no = 924 # use last 3 matric digits
    #lr_list = [0.1, 0.01, 0.001, 0.00001]
    #num_iters = 20000

    #---------------------------------------PART A------------------------------------------

    # build X with bias (no normalization case)
    X_train_poly = np.hstack([FS_X_train, np.ones((FS_X_train.shape[0], 1))])
    X_val_poly   = np.hstack([FS_X_val,   np.ones((FS_X_val.shape[0],   1))])
    X_test_poly  = np.hstack([FS_X_test,  np.ones((FS_X_test.shape[0],  1))])

    acc_train_list_Log, acc_val_list_Log = [], []
    cost_dict = {}
    best_lr, best_val_acc, W_best = None, -1.0, None

    for lr in lr_list:
        np.random.seed(seed_matrix_no)
        W0 = np.random.normal(0, 0.1, (X_train_poly.shape[1], Ytr_onehot.shape[1]))

        W_opt, cost_vec, Ytr_est = multinomial_logistic_regression(X_train_poly, W0, Ytr_onehot, lr, num_iters)
        cost_dict[lr] = cost_vec

        # training accuracy (inline)
        ytr_pred = np.argmax(Ytr_est, axis=1)
        ytr_true = np.argmax(Ytr_onehot, axis=1)
        train_acc = np.mean(ytr_pred == ytr_true)

        # validation accuracy: **use X_val_poly**
        Yval_est, _, _ = multi_logistic_cost_gradient(X_val_poly, W_opt, Yval_onehot)
        yval_pred = np.argmax(Yval_est, axis=1)
        yval_true = np.argmax(Yval_onehot, axis=1)
        val_acc = np.mean(yval_pred == yval_true)

        acc_train_list_Log.append(train_acc)
        acc_val_list_Log.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr
            W_best = W_opt

    # test accuracy ONCE with best W on X_test_poly
    Yts_est, _, _ = multi_logistic_cost_gradient(X_test_poly, W_best, Yts_onehot)
    yts_pred = np.argmax(Yts_est, axis=1)
    yts_true = np.argmax(Yts_onehot, axis=1)
    test_acc_Log = np.mean(yts_pred == yts_true)

    #---------------------------------------PART B------------------------------------------

    scaler = StandardScaler()
    Xtr_n  = scaler.fit_transform(FS_X_train)   # fit on TRAIN only
    Xval_n = scaler.transform(FS_X_val)
    Xts_n  = scaler.transform(FS_X_test)

    X_tr_n  = np.hstack([Xtr_n,  np.ones((Xtr_n.shape[0],  1))])
    X_val_n = np.hstack([Xval_n, np.ones((Xval_n.shape[0], 1))])
    X_ts_n  = np.hstack([Xts_n,  np.ones((Xts_n.shape[0],  1))])

    cost_dict_norm = {}
    acc_train_list_Log_norm, acc_val_list_Log_norm = [], []
    best_lr_norm, best_val_acc_norm, W_best_norm = None, -1.0, None

    for lr in lr_list:
        np.random.seed(seed_matrix_no)
        W0n = np.random.normal(0, 0.1, (X_tr_n.shape[1], Ytr_onehot.shape[1]))

        Wn_opt, cost_vec_n, Ytr_est_n = multinomial_logistic_regression(X_tr_n, W0n, Ytr_onehot, lr, num_iters)
        cost_dict_norm[lr] = cost_vec_n

        ytr_pred_n = np.argmax(Ytr_est_n, axis=1)
        ytr_true   = np.argmax(Ytr_onehot, axis=1)
        train_acc_n = np.mean(ytr_pred_n == ytr_true)

        Yval_est_n, _, _ = multi_logistic_cost_gradient(X_val_n, Wn_opt, Yval_onehot)
        yval_pred_n = np.argmax(Yval_est_n, axis=1)
        yval_true   = np.argmax(Yval_onehot, axis=1)
        val_acc_n = np.mean(yval_pred_n == yval_true)

        acc_train_list_Log_norm.append(train_acc_n)
        acc_val_list_Log_norm.append(val_acc_n)

        if val_acc_n > best_val_acc_norm:
            best_val_acc_norm = val_acc_n
            best_lr_norm = lr
            W_best_norm = Wn_opt

    Yts_est_n, _, _ = multi_logistic_cost_gradient(X_ts_n, W_best_norm, Yts_onehot)
    yts_pred_n = np.argmax(Yts_est_n, axis=1)
    yts_true   = np.argmax(Yts_onehot, axis=1)
    test_acc_Log_norm = np.mean(yts_pred_n == yts_true)

    # return in this order      

    return cost_dict,acc_train_list_Log, acc_val_list_Log, best_lr,test_acc_Log, cost_dict_norm, acc_train_list_Log_norm, acc_val_list_Log_norm, best_lr_norm, test_acc_Log_norm

def cost_vs_iter_curve (cost_dict, cost_dict_norm):
    """
    Args:
        cost_dict (dict): Dictionary of cost values for each learning rate without input normalization.
        cost_dict_norm (dict): Dictionary of cost values for each learning rate with input normalization.

    """

    # your code goes here

    # Figure 1: without normalization
    plt.figure(figsize=(9, 4.5))
    plt.rcParams.update({'font.size': 12})
    for lr in sorted(cost_dict.keys()):
        costs = cost_dict[lr]
        plt.plot(np.arange(len(costs)), costs, label=f"lr={lr}")
    plt.xlabel("Iterations")
    plt.ylabel("Mean categorical cross-entropy J(W)")
    plt.title("Multinomial Logistic Regression — no normalization")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure 2: with z-score normalization
    plt.figure(figsize=(9, 4.5))
    plt.rcParams.update({'font.size': 12})
    for lr in sorted(cost_dict_norm.keys()):
        costs = cost_dict_norm[lr]
        plt.plot(np.arange(len(costs)), costs, label=f"lr={lr}")
    plt.xlabel("Iterations")
    plt.ylabel("Mean categorical cross-entropy J(W)")
    plt.title("Multinomial Logistic Regression — z-score normalization")
    plt.legend()
    plt.tight_layout()
    plt.show()


cost_dict,acc_train_list_Log, acc_val_list_Log, best_lr,test_acc_Log, cost_dict_norm, acc_train_list_Log_norm, acc_val_list_Log_norm, best_lr_norm, test_acc_Log_norm = MLR_select_lr(FS_X_train, FS_X_val, FS_X_test, Ytr_onehot, Yval_onehot, Yts_onehot)

print(f"Without Normalization")
print(f"Training accuracies for different learning rates: {np.round(acc_train_list_Log,2)}")
print(f"Validation accuracies for different learning rates: {np.round(acc_val_list_Log,2)}")
print(f"Best learning rate: {best_lr}")
print(f"Test accuracy for best learning rate {best_lr}: {np.round(test_acc_Log,2)}")


print(f"With Z-score Standardization")
print(f"Training accuracies for different learning rates: {np.round(acc_train_list_Log_norm,2)}")
print(f"Validation accuracies for different learning rates: {np.round(acc_val_list_Log_norm,2)}")
print(f"Best learning rate: {best_lr_norm}")
print(f"Test accuracy for best learning rate {best_lr_norm}: {np.round(test_acc_Log_norm,2)}")

cost_vs_iter_curve (cost_dict, cost_dict_norm)
### Analysis of Effect of Normalization Based on Your Results