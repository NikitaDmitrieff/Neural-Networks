import torch
import torch.nn as nn
from numpy.random import default_rng
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt
import torch.optim as optim
# from skorch import NeuralNetRegressor
#import tqdm
# from sklearn.model_selection import GridSearchCV


class LinearRegression(nn.Module):
    def __init__(self, n_input_vars, n_output_vars, nb_hidden):

        if nb_hidden == 3:
            hidden_size = [256, 512, 128]
        elif nb_hidden == 4:
            hidden_size = [256, 512, 256, 128]
        elif nb_hidden == 5:
            hidden_size = [256, 512, 512, 256, 128]
        elif nb_hidden == 6:
            hidden_size = [256, 512, 1024, 512, 256, 128]

        super().__init__()  # call constructor of superclass
        self.input_layer = nn.Linear(n_input_vars, hidden_size[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size[i], hidden_size[i + 1]) for i in range(len(hidden_size) - 1)])
        self.output_layer = nn.Linear(hidden_size[-1], n_output_vars)

    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))

        return self.output_layer(x)


def split_dataset(x, y, test_proportion, random_generator=default_rng()):
    """ Split dataset into training and test sets, according to the given
        test set proportion.

    Args:
        x (pd.DataFrame): Instances, numpy array with shape (N,K)
        y (pd.DataFrame): Output label, numpy array with shape (N,)
        test_proportion (float): the desired proportion of test examples
                                 (0.0-1.0)
        random_generator (np.random.Generator): A random generator

    Returns:
        tuple: returns a tuple of (x_train, x_test, y_train, y_test)
               - x_train (pd.DataFrame): Training instances shape (N_train, K)
               - x_test (pd.DataFrame): Test instances shape (N_test, K)
               - y_train (pd.DataFrame): Training labels, shape (N_train, )
               - y_test (pd.DataFrame): Test labels, shape (N_test, )
    """

    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test

    x_train = x.loc[shuffled_indices[:n_train]]
    y_train = y.loc[shuffled_indices[:n_train]]
    x_test = x.loc[shuffled_indices[n_train:]]
    y_test = y.loc[shuffled_indices[n_train:]]

    return x_train, x_test, y_train, y_test


class Regressor():

    #def __init__(self, x, nb_epoch=1000, nb_batch=256, nb_hidden=6):
    def __init__(self, x, nb_epoch=1000):
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
        #self.nb_batch = nb_batch
        self.nb_batch = 128

        self.X, self.Y = self._preprocessor(x, training=True)
        self.input_size = self.X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.labelB = LabelBinarizer()

        #self.model = MutliLinearRegression(n_input_vars=self.input_size, n_output_vars=1, nb_hidden=nb_hidden)
        self.model = LinearRegression(n_input_vars=self.input_size, n_output_vars=1, nb_hidden=6)

        self.criterion = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
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

        clean_x = final_df.fillna(final_df['total_bedrooms'].median())
        normalized_x = (clean_x - clean_x.min()) / (clean_x.max() - clean_x.min())

        if y is not None:
            y = torch.tensor(y.astype(np.float32).values)
            # assert len(clean_x.index) == len(y.index)

        X_torch_tensor = torch.tensor(normalized_x.astype(np.float32).values)

        return X_torch_tensor, (y if training else None)
        # save settings

        # print(normalized_x.columns)
        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

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

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget

        for epoch in range(self.nb_epoch):
            # Reset the gradients
            self.optimiser.zero_grad()
            # forward pass
            y_hat = self.model(X)
            # compute loss
            loss = self.criterion(y_hat, Y)
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

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        y_predictions = self.model(X)

        return y_predictions.detach().numpy()

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

        mse = mean_squared_error(y, y_predictions, squared=True)
        SSres = np.sum((y.to_numpy()-y_predictions)**2)
        SStot = ((y.to_numpy() - y.to_numpy().mean()) ** 2).sum()
        determination_coef = 1 - SSres / SStot

        print('The R2 is: ', determination_coef)
        rmse = mean_squared_error(y, y_predictions, squared=False)
        print('The RMSE is: ', rmse)

        return rmse

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


def RegressorHyperParameterSearch(x_train, x_test, y_train, y_test):
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape
            (batch_size, input_size).
        - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

    Returns:
        The function should return your optimised hyper-parameters.

    """
    return
    # parameters_dic = {'nb_batch': 1024, 'nb_epoch': 10, 'nb_hidden': 3}

    # regressor = Regressor(x_train,
    #                       nb_epoch=parameters_dic['nb_epoch'],
    #                       nb_batch=parameters_dic['nb_batch'],
    #                       nb_hidden=parameters_dic['nb_hidden'])

    # regressor.fit(x_train, y_train)
    # lowest_error = regressor.score(x_test, y_test)

    # parameters = {'nb_hidden': [3, 4, 5, 6],
    #               'nb_epoch': [50, 100, 500],
    #               'nb_batch': [128, 256, 512],
    #               }

    # batchs = [[],[],[],[]]
    # epochs = [[],[],[],[]]
    # hiddens = [[],[],[],[]]
    # errors = [[],[],[],[]]

    # for batch in parameters['nb_batch']:
    #     for epoch in parameters['nb_epoch']:
    #         for idx, hidden in enumerate(parameters['nb_hidden']):

    #             mean_error = cross_val(x_train, y_train, nb_epoch=epoch, nb_batch=batch, nb_hidden=hidden, cv=5)

    #             batchs[idx].append(batch)
    #             epochs[idx].append(epoch)
    #             hiddens[idx].append(hidden)
    #             errors[idx].append(mean_error)

    #             if mean_error < lowest_error:
    #                 lowest_error = mean_error
    #                 parameters_dic["nb_hidden"] = hidden
    #                 parameters_dic["nb_epoch"] = epoch
    #                 parameters_dic["nb_batch"] = batch



    # # Creating figure
    # #fig = plt.figure(figsize=(10, 7))
    # #ax = plt.axes(projection="3d")
    # #ax.set_xlabel('Batch size')
    # #ax.set_ylabel('Number of epochs')
    # #ax.set_zlabel('Error (RMSE)')

    # # for hidden in hiddens:
    # #ax.scatter3D(batchs[0], epochs[0], errors[0], marker='<')
    # #ax.scatter3D(batchs[1], epochs[1], errors[1], marker='o')
    # #ax.scatter3D(batchs[2], epochs[2], errors[2], marker='x')
    # #ax.scatter3D(batchs[3], epochs[3], errors[3], marker='s')
    # #ax.legend(['3','4','5','6'], title='Number of hidden layers', loc='best')

    # #ax.grid(True)
    # #plt.title("Regressor Hyperparameter Search between 36 models with Adam optimizer, \n learning rate = 1e-4 and 5 folds cross validation")
    # # show plot
    # #plt.show()

    # return parameters_dic, lowest_error
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


# def cross_val(x, y, nb_epoch, nb_batch, nb_hidden, cv=5):

#     list_of_errors = []
#     kf = KFold(n_splits=cv)
#     kf.get_n_splits(x)

#     for train_index, val_index in kf.split(x):

#         x_train, x_val = x.iloc[train_index], x.iloc[val_index]
#         y_train, y_val = y.iloc[train_index], y.iloc[val_index]

#         regressor = Regressor(x_train, nb_epoch=nb_epoch, nb_batch=nb_batch, nb_hidden=nb_hidden)
#         regressor.fit(x_train, y_train)
#         error = regressor.score(x_val, y_val)
#         list_of_errors.append(error)

#     avg = sum(list_of_errors) / len(list_of_errors)

#     return avg





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

    regressor = Regressor(x_train, nb_epoch=50)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))



if __name__ == "__main__":
    example_main()
