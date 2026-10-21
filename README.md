# Deep Learning Optimization Methods (Coursera)

This repository contains the implementation of various optimization algorithms and learning rate scheduling techniques in Python (using NumPy), built as part of the Deep Learning Specialization (Course 2: Improving Deep Neural Networks).

##  Implemented Features & Exercises

1. **Gradient Descent (GD):**
   - Implemented standard batch gradient descent update rules.
2. **Mini-batch Gradient Descent & Shuffling:**
   - Implemented random mini-batch generation (`random_mini_batches`) involving data shuffling and partitioning.
3. **Momentum Optimization:**
   - Initialized velocity variables (`initialize_velocity`).
   - Implemented parameter updates using Exponentially Weighted Averages (Momentum).
4. **Adam Optimization:**
   - Initialized first and second moment tracking variables (`initialize_adam`).
   - Implemented the complete Adam update rule including bias correction for $v$ and $s$.
5. **Learning Rate Scheduling:**
   - Exponential weight decay (`update_lr`).
   - Fixed interval scheduling for learning rate decay (`schedule_lr_decay`).

##  Project Structure

* `Optimization_methods.ipynb`: The main Jupyter Notebook containing all implemented functions, test cases, and model trainings.
* `opt_utils_v1a.py`: Helper functions for forward/backward propagations, dataset loading, and plotting decision boundaries.
* `public_tests.py` & `testCases.py`: Unit test files to validate the correctness of each implemented exercise.
* `datasets/`: Contains the synthetic dataset used for training and testing optimization algorithms.
* `images/`: Image assets used in the notebook.

##  Requirements

To run this notebook locally, ensure you have the following packages installed:
* Python 3.x
* NumPy
* Matplotlib
