import numpy as np
from relight.relight import relight

results = np.zeros((1, 2))
masked = False
model_path = 'test_dataset/outputs/Student'
test_data = 'test_dataset/test_fig6'
relight(model_path, test_data)