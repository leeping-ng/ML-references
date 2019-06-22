# AI Apprenticeship Programme - Housing Prices in Sindian District, Taiwan

**1. How to Run the Code**
- Check that all required libraries listed in requirements.txt are installed
- Open a terminal, and cd to the base folder containing run.sh
- Run the bash script using ./run.sh
- The machine learning pipeline will run, based on the chosen configuration in config.py
- For more details on how to change the model, parameters, or pre-processing option in config.py, refer to Section 2.1 below.

**2. Program Design**
- run.sh is a bash script which, when executed, runs the main.py program in the folder named mlp.

The mlp folder contains the following files:

**2.1 config.py**
- This is the configuration file where the configurable parameters are stored and can be modified
- *bin_house_age*: Set to True if you want to sort the 'X2 house age' feature into bins for the machine learning pipeline to process. Set to False if not.
- *test_train_ratio*: The ratio of the test to training dataset, which should be a value between 0 and 1. I usually set it to 0.25.
- *cv_folds*: The number of cross-validation folds to use. I usually set it to 5.
- *model_name*: One of the following 6 machine learning models can be selected. You can copy and paste from the examples in config.py. 
  - 'LinearRegression'
  - 'Lasso' 
  - 'Ridge'
  - 'RandomForestRegressor'
  - 'AdaBoostRegressor'
  - 'GradientBoostingRegressor'
- *grid_search_params*: The range of parameters to perform grid search on. You can copy and paste from the examples in config.py.
  - If adding a new parameter such as 'min_samples_split', ensure that 'model__' precedes this, in other words: 'model__min_samples_split' should be the name of the key

**2.2 utils.py**
- This is the file where the following helper functions are stored
- *get_model()*: This function takes in the model_name from config.py, imports the model from the right library, and returns the imported model to main.py
- *bin_house_age()*: This function takes in the dataframe and bins the 'X2 house age' feature, returning a dataframe with the binned features to main.py
  - This function is also called in the Jupyter Notebook for Task 1 (EDA). When running the Jupyter Notebook, ensure that the path environment variable is set to utils.py, to be able to use this function

**2.3 main.py**
- This is the main program that will be executed by run.sh
- The helper functions from utils.py and variables defined in config.py are called here
- The machine learning pipeline will read the data from the URL, pre-process the data, train the selected model, and output the Root Mean Squared Error (RMSE) of the model
