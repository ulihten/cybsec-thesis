# Create a simple machine learning model with supervised learning algorithm (Linear regression)
# We will create a training data set of psuedorandom numbers as input and our own function for the training data set output using the generated numbers as input. 
# Then we will train our model on the newly created dataset. And test the model and compare prediction to the expected output. 
# Function
# f(a,b,c) = 8a + 4b + 3c = y

# Credits: https://medium.com/@randerson112358/a-simple-machine-learning-python-program-bf5d156d2cda
# made modifications in the function and added possibility for the user to enter the data and let the script predict the outcome of the function.

# Import libraries
from random import randint
from sklearn.linear_model import LinearRegression

# Range limit of random numbers and number of rows in training data
TRAIN_LIMIT = 2500
TRAIN_ROW_COUNT = 1000

# Empty lists for training data input and corresponding output
TRAINING_INPUT = list()
TRAINING_OUTPUT = list()

# Create and append generated data set to input and output lists
for i in range(TRAIN_ROW_COUNT):
    a = randint(0, TRAIN_LIMIT)
    b = randint(0, TRAIN_LIMIT)
    c = randint(0, TRAIN_LIMIT)
    # Create linear function for the output dataset Y
    output = (8 * a) + (4 * b) + (3 * c)
    TRAINING_INPUT.append([a,b,c])
    TRAINING_OUTPUT.append(output)

# Create linear regression object (n_jobs -1 means using all processors)
predictor = LinearRegression(n_jobs=-1)

# fit linear model
predictor.fit(X=TRAINING_INPUT,y=TRAINING_OUTPUT,)

# Create a testing data set. Output should be 270
X_TEST_DATA_SET = [[10,20,30]]

# Predict output of test data set
outcome = predictor.predict(X=X_TEST_DATA_SET)

# The estimated coefficents for the linear regression problem
coefficents = predictor.coef_ 

print('Function f(a,b,c) = 8a + 4b + 3c')
print('Test data set: a = 10, b = 20, c = 30')
print('Outcome: {} \nCoefficents: {}'.format(outcome, coefficents))

print('Time for prediction for user inputted data set \n Insert A')
X_USER_DATA_SET_A = int(input())

print ('Insert B')
X_USER_DATA_SET_B = int(input())

print ('Insert C')
X_USER_DATA_SET_C = int(input())

X_USER_DATA_SET = [[X_USER_DATA_SET_A,X_USER_DATA_SET_B,X_USER_DATA_SET_C]]

user_outcome = predictor.predict(X = X_USER_DATA_SET)
user_coefficents = predictor.coef_

print('Outcome: {} \nCoefficents: {}'.format(user_outcome, user_coefficents))