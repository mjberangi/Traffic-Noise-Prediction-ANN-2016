from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

def get_ann_model():
    """
    Returns the ANN model with the desired configurations in Table 6 of the paper.
    
    Structure: 6 Inputs -> 10 Hidden Neurons -> 1 Output
    Activation: Sigmoid (Logistic) 
    Algorithm: Levenberg-Marquardt (approximated by 'lbfgs') 
    """
    model = MLPRegressor(
        hidden_layer_sizes=(10,),
        activation='logistic',  
        solver='lbfgs',         
        max_iter=1000,
        random_state=42,
        alpha=0.0001
    )
    return model

def get_regression_model():
    """
    Returns standard Linear Regression for comparison
    """
    return LinearRegression()
