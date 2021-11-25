import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import *

data = sklearn.datasets.load_wine(return_X_y=False, as_frame=True)
data =data.frame
print(data)
