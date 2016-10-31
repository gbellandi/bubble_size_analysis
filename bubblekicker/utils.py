
import numpy as np


def calculate_circularity_reciprocal(perimeter, area):
    """calculate the circularity based on the perimeter and area"""
    return (perimeter**2)/(4*np.pi*area)


def calculate_convexity(perimeter, area):
    """calculate the circularity based on the perimeter and area"""
    return area/perimeter
