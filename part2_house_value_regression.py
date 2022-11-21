import torch
import torch.nn as nn
from numpy.random import default_rng
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error


class LinearRegression(nn.Module):
    def __init__(self, n_input_vars, n_output_vars=1):
        super().__init__() # call constructor of superclass
        self.linear = nn.Linear(n_input_vars, n_output_vars)

    def forward(self, x):
        return self.linear(x)


def split_dataset(x, y, test_proportion, random_generator=default_rng()):
    """ Split dataset into training and test sets, according to the given
        test set proportion.

    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Output label, numpy array with shape (N,)
        test_proprotion (float): the desired proportion of test examples
                                 (0.0-1.0)
        random_generator (np.random.Generator): A random generator

    Returns:
        tuple: returns a tuple of (x_train, x_test, y_train, y_test)
               - x_train (np.ndarray): Training instances shape (N_train, K)
               - x_test (np.ndarray): Test instances shape (N_test, K)
               - y_train (np.ndarray): Training labels, shape (N_train, )
               - y_test (np.ndarray): Test labels, shape (N_test, )
    """

    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test

    x_train = x.loc[shuffled_indices[:n_train]]
    y_train = y.loc[shuffled_indices[:n_train]]
    x_test = x.loc[shuffled_indices[n_train:]]
    y_test = y.loc[shuffled_indices[n_train:]]


    return (x_train, x_test, y_train, y_test)

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.X, self.Y = self._preprocessor(x, training = True)
        self.input_size = self.X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.labelB = LabelBinarizer()
        self.model = LinearRegression(n_input_vars=self.input_size)
        self.criterion = torch.nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=0.0001)

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        for column in x:
            is_num = is_numeric_dtype(x[column])

            if not is_num:
                if training:
                    self.labelB = LabelBinarizer().fit_transform(x[[column]])
                    encoder_df = pd.DataFrame(self.labelB)
                    final_df = x.join(encoder_df)
                    final_df.drop(column, axis=1, inplace=True)
                else:
                    encoder_df = pd.DataFrame(self.labelB)
                    final_df = x.join(encoder_df)
                    final_df.drop(column, axis=1, inplace=True)


        clean_x = final_df.fillna(1)

        # if y is not None:
        #     removed_indexes = x[~x.index.isin(clean_x.index)]
        #     removed_indexes_list = list(removed_indexes.index.values)
        #     clean_y = y.drop(removed_indexes_list)

        normalized_x = (clean_x - clean_x.min()) / (clean_x.max() - clean_x.min())

        if y is not None:
            assert len(clean_x.index) == len(y.index)
        # save settings

        # print(normalized_x.columns)
        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        return normalized_x, (y if training else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        X_torch_tensor = torch.tensor(X.astype(np.float32).values)
        Y_torch_tensor = torch.tensor(Y.astype(np.float32).values)


        for epoch in range(self.nb_epoch):
            # Reset the gradients
            self.optimiser.zero_grad()
            # forward pass
            y_hat = self.model(X_torch_tensor)
            # compute loss
            loss = self.criterion(y_hat, Y_torch_tensor)
            # Backward pass (compute the gradients)
            loss.backward()
            # update parameters
            self.optimiser.step()

            # print(f"Epoch: {epoch}\t w: {self.model.linear.weight.data[0]}\t b: {self.model.linear.bias.data[0]:.4f} \t L: {loss:.4f}")

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False) # Do not forget
        X_torch_tensor = torch.tensor(X.astype(np.float32).values)
        y_predictions = self.model.forward(X_torch_tensor)

        return y_predictions.tolist()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        y_predictions = self.predict(x)

        assert len(y) == len(y_predictions)
        return mean_squared_error(y, y_predictions, squared=False)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    seed = 60012
    rg = default_rng(seed)

    x_train, x_test, y_train, y_test = split_dataset(x, y,
                                                     test_proportion=0.2,

                                                     random_generator=rg)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting

    regressor = Regressor(x_train, nb_epoch = 100000)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

