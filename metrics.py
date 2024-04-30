import numpy as np

def mape(y_true, y_pred):
    # Calculate absolute percentage error for each element
    abs_percentage_error = np.abs((y_true - y_pred) / y_true) * 100
    # Calculate mean and standard deviation of MAPE
    mean_mape = np.mean(abs_percentage_error)
    std_mape = np.std(abs_percentage_error)
    return mean_mape, std_mape, abs_percentage_error

def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)