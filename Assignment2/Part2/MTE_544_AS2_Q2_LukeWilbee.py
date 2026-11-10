"""
Student Name & Last Name: Luke Wilbee
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q2_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

# https://www.youtube.com/watch?v=H7EwHBN9Vus citation for positive definite
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pandas as pd


def plot_ellipse(S, b, ax):
    #lambda_var = 9.80665         ## Could not get fitted elipse to plot made update below
    #eigvals, eigvecs = np.linalg.eigh(lambda_var*S)
    #aa, bb = eigvals
    eigvals, eigvecs = np.linalg.eigh(S)
    aa, bb = 1/np.sqrt(eigvals)

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse_param = np.array([aa * unit_circle[0], bb * unit_circle[1]])
    ellipse_points = eigvecs @ ellipse_param + b[:, np.newaxis]

    ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-', label='Fitted Ellipse')
    ax.plot(b[0], b[1], 'ro', label='Center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    ax.grid(True)

def visualize_data(p, ax, inliers, threshold):
    ax.scatter(p[:, 0], p[:, 1], color='red', alpha=0.5, label='Raw Measurements (Ellipse)')
    ax.scatter(inliers[:, 0], inliers[:, 1], color='purple', alpha=0.7, label='Inliers')

    for point in inliers:
        circle = plt.Circle(point, threshold, color='orange', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(circle)

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def fit_ellipse_subset(points):
    # X and Y coordinates of points
    x = points[:, 0]
    y = points[:, 1]
    # 5x5 matrix of system
    M = np.column_stack([x**2, 2*x*y, y**2, -2*x, -2*y])
    # Right hand constants matrix
    r = -1*np.ones(5)
    # Solve for coefficients
    A, B, C, D, E = np.linalg.solve(M, r)
    # Solve for b and Q
    P = np.array([[A, B], [B, C]])
    d = np.array([D, E])
    Pinv = np.linalg.inv(P)
    alpha = 1.0 / (np.transpose(d) @ Pinv @ d - 1.0)
    Q = alpha * P
    b = Pinv @ d
    # Check positive definite
    #eigvals = np.linalg.eigvalsh(Q)
    #print("eigenvalues of Q:", eigvals, "-> ellipse" if np.all(eigvals > 0) else "-> NOT an ellipse")

    return Q, b

def ransac_ellipse(data, num_iterations=1000, threshold=0.2):
    inliers = []
    # Given the data sets, perform RANSAC to find the best Q and b as well as the inliers
    # Hint: You should use fit_ellipse_subset 
    # Hint: in some case, the Q matrix might not be positive defintie, use is_positive_definite to check.
    
    ##### ADD your code here : #####
    n = data.shape[0]
    Q, b = None, None
    bestCount = 0
    lo, hi = data.min(axis=0), data.max(axis=0) #For b position correction

    for j in range(num_iterations):
        # Obtain 5 random data points
        ranIdx = np.random.choice(n, 5, replace=False)
        sPoints = data[ranIdx]
        # Fit an ellipse to random points
        try:
            Q_i, b_i = fit_ellipse_subset(sPoints)
        except np.linalg.LinAlgError:
            continue
        # Check if Q_i is positive definite
        if not is_positive_definite(Q_i):
            continue
        # Check if found b is in reasonable position 
        # (fixes issue with obnormal center location)
        if not (np.all(b_i >= lo) and np.all(b_i <= hi)):
            continue
        # Find number of inliers
        diff = data - b_i                                   
        d = np.sqrt(np.einsum('ij,jk,ik->i', diff, Q_i, diff))
        mask = np.abs(d - 1.0) <= threshold
        count = np.sum(mask)
        # Save best result
        if count > bestCount:
            bestCount = count
            Q, b = Q_i, b_i
            inliers = data[mask]
    ##### END #####
    return Q,b,inliers

if __name__ == "__main__":
    # Load the data from CSV file and select N random points
    N = 1000
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]
    # dataset is p
    
    Q, b_est, inliers = ransac_ellipse(dataset, 1000, 0.1)
    
    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=0.5)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    plt.show()
