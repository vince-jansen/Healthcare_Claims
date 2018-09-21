# This script grabs the data, formats the data, and measures the accuracy of three different model types on an out of sample set
# This script plays with an OLS model, a Tensorflow neural network, and a pyrenn neural network
import numpy as np
import sklearn
import math
import statsmodels.api as sm
import matplotlib.pyplot as pl
import keras
import pyrenn

# Importing the raw data as strings
raw_data_from_csv = np.loadtxt("Healthcare_Claims.csv", delimiter=",", dtype='str')

# Changing sex and smoker values from strings to binary values
raw_data_from_csv[raw_data_from_csv == 'female'] = '0'
raw_data_from_csv[raw_data_from_csv == 'male'] = '1'
raw_data_from_csv[raw_data_from_csv == 'no'] = '0'
raw_data_from_csv[raw_data_from_csv == 'yes'] = '1'

# Creating an array with three columns
temp_location = np.zeros((raw_data_from_csv.shape[0],raw_data_from_csv.shape[1]-4))

# The first column will be a 1 if from the northeast, the second if from the northwest, the third if from the southeast, and all zeros if from the southwest
for i in range(1,raw_data_from_csv.shape[0]-1):
    if raw_data_from_csv[i,5] == 'northeast':
        temp_location[i,0] = 1
    elif raw_data_from_csv[i,5] == 'northwest':
        temp_location[i,1] = 1
    elif raw_data_from_csv[i,5] == 'southeast':
        temp_location[i,2] = 1

# formatted_data is an array with the three location columns of data and the remaining variables converted to float data        
formatted_data = np.column_stack((temp_location[1:,:],raw_data_from_csv[1:,[0,1,2,3,4,6]].astype(np.float)))

# The explanatory variables are put in X_data and the response variable is put in Y_data
X_data = formatted_data[:,0:8]
Y_data = formatted_data[:,8]

# The sample is split in two, the first of which will create the model, the second of which will measure out of sample accuracy
X_train = X_data[0:math.ceil(X_data.shape[0]/2),:]
Y_train = Y_data[0:math.ceil(Y_data.shape[0]/2),]
X_test = X_data[math.ceil(X_data.shape[0]/2):,:]
Y_test = Y_data[math.ceil(Y_data.shape[0]/2):,]

# A constant is added to the data for a linear fit
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# The training data is fit via OLS
# t values are low for some variables but removing these variables from the OLS model doesn't increase performance on out of sample data
linear_model = sm.OLS(Y_train,X_train).fit()
print(linear_model.summary())
linear_predictions = linear_model.predict(X_test)
# A scatter of OLS model claims predictions vs. actual claims
pl.scatter(linear_predictions.T,Y_test.T)
pl.show()

# Data is scaled for neural network training
scaler = sklearn.preprocessing.StandardScaler().fit(X_train[:,1:])
X_train = scaler.transform(X_train[:,1:])
X_test = scaler.transform(X_test[:,1:])

# A neural network via Keras on Tensorflow. 
# Using hidden layers with 40 neurons each, 'relu' activation functons, 'adam' optimizer measured via 'mse'
# A validation set uses 35% of the training data to prevent overfitting
# The predictions are averaged across many different models with different training and validation sets
for i in range(0,10):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(40, input_dim=8, activation='relu'))
    model.add(keras.layers.Dense(40, activation='relu'))
    model.add(keras.layers.Dense(40, activation='relu'))
    model.add(keras.layers.Dense(40, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
	
    model.fit(X_train, Y_train, epochs=1000, verbose=0, callbacks=[earlystopping], validation_split=0.35, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=1, validation_steps=5)
	
    if i == 0:
        nn_predictions = model.predict(X_test)
    else:
        nn_predictions = (nn_predictions*(i)+model.predict(X_test))/(i+1)

    print(i)
    # Prints the correlation between the average of model outputs and the targets and then the correlation between the most recent model output and the targets
    print(np.corrcoef(nn_predictions.T,Y_test.T)[0,1]**2)
    print(np.corrcoef(model.predict(X_test).T,Y_test.T)[0,1]**2)

# A scatter of Tensorflow model claims predictions vs. actual claims
pl.scatter(nn_predictions.T,Y_test.T)
pl.show()

# Creating a neural network using the Levenberg-Marquardt backpropagation training function
# Used for quick descent training and possibly a more accurate prediction
# Fewer hidden layers and less nodes are used due to a larger propensity to overfit
# Cannont use a validation set for early stopping in pyrenn so these two lines are used to find convergence
# Seems to converge around 10 epoches. Should stop early at 10 epoches to avoid overfitting on a small dataset
net = pyrenn.CreateNN([8,5,1])
pyrenn.train_LM(X_train.T, Y_train.T, net,verbose=1,k_max=20)

# The predictions are averaged across many different trained models
for i in range(0,10):
    print(i)
    net = pyrenn.CreateNN([8,5,1])
    pyrenn.train_LM(X_train.T, Y_train.T, net,verbose=0,k_max=10)
    if i == 0:
        LM_predictions = pyrenn.NNOut(X_test.T, net)
    else:
        LM_predictions = (LM_predictions*(i)+pyrenn.NNOut(X_test.T, net))/(i+1)
    
    print(i)
    # Prints the correlation between the average of model outputs and the targets and then the correlation between the most recent model output and the targets
    print(np.corrcoef(LM_predictions.T,Y_test.T)[0,1]**2)
    print(np.corrcoef(pyrenn.NNOut(X_test.T, net).T,Y_test.T)[0,1]**2)
# A scatter of Pyrenn model claims predictions vs. actual claims
pl.scatter(LM_predictions.T,Y_test.T)
pl.show()

# Accuracy measures for each model
# The linear model R^2 = 75.15%, MAE = $4,165
# The Tensorflow model R^2 ~ 81.0%, MAE ~ $3,100
# The Pyrenn model R^2 ~ 84.5%, MAE ~ $2,900
print('The OLS model R^2 is' ,np.corrcoef(linear_predictions.T,Y_test.T)[0,1]**2)
print('The OLS model MAE is' ,np.mean(np.abs(linear_predictions.T-Y_test.T)))
print('The Tensorflow model R^2 is' ,np.corrcoef(nn_predictions.T,Y_test.T)[0,1]**2)
print('The Tensorflow model MAE is' ,np.mean(np.abs(nn_predictions.T-Y_test.T)))
print('The Pyrenn model R^2 is' ,np.corrcoef(LM_predictions.T,Y_test.T)[0,1]**2)
print('The Pyrenn model MAE is' ,np.mean(np.abs(LM_predictions.T-Y_test.T)))

# Accuracy measurements vary when only averaging over 10 trained models. Averaging over 100+ trained models will be more consistent
# This small dataset causes overfitting easily and hinders accuracy. The easiest way to increase the accuracy of each model would be to add to the sample size
# The neural networks are 5% to 10% more accurate!