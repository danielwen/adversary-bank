"""
Functions for computing Equal Error Rate (EER)
"""

import numpy as np

def ternary_search(difference, left, right, tol, max_iter=50):
    iters = 0

    while True:
        if abs(right - left) < tol:
            return (left + right)/2

        left_third = left + (right - left)/3
        right_third = right - (right - left)/3

        left_diff = difference(left_third)
        right_diff = difference(right_third)

        if left_diff == 0:
            return left_third
        elif right_diff == 0:
            return right_third
        elif left_diff < 0 and right_diff > 0:
            left = left_third
            right = right_third
        elif left_diff < 0 and right_diff < 0:
            left = right_third
        elif left_diff > 0 and right_diff > 0:
            right = left_third
        else:
            raise Exception("Ill function")
        
        iters += 1
        if iters >= max_iter:
            raise Exception("Ternary search did not converge")

def compute_accept_rate(scores, threshold):
    return np.count_nonzero(scores > threshold) / scores.size

def compute_eer(user_scores, attack_scores, min_thresh, max_thresh, tol=0.01):
    def difference(threshold):
        user_reject_rate = 1 - compute_accept_rate(user_scores, threshold)
        attack_accept_rate = compute_accept_rate(attack_scores, threshold)
        return user_reject_rate - attack_accept_rate
    
    threshold = ternary_search(difference, min_thresh, max_thresh, tol)

    return (threshold, compute_accept_rate(user_scores, threshold),
        compute_accept_rate(attack_scores, threshold))
