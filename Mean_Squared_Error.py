import numpy as np

class Mean_Squared_Error:

    def mse(y_pred, y_true):
        # 1/n * ((y_pred - y_true) ** 2)
        return np.mean( (y_pred - y_true) ** 2)
    
    def mse_derivate(y_pred, y_true):
        # 2/n * (y_pred - y_true)
        return (2 / len(y_true)) * (y_pred - y_true)

