#!/usr/bin/env python
# coding: utf-8

# In[2]:


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas import optim

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend

import QuantLib as ql
import numpy as np

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    mse = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_mse = np.argmin(mse)
    best_trial_obj = valid_trial_list[index_having_minumum_mse]
    return best_trial_obj['result']['model']

class Option:
    def __init__(self, calculation_date, maturity, stock_price, strike_price, volatility, dividend_rate, risk_free_rate, option_type):
        self.maturity = maturity
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.dividend_rate = dividend_rate
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type
        self.calculation_date = calculation_date
        self.bs_price = -1
        
    def BSM_price(self):
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates()
        ql.Settings.instance().evaluationDate = self.calculation_date
        
        payoff = ql.PlainVanillaPayoff(self.option_type, self.strike_price)
        exercise = ql.EuropeanExercise(self.maturity)
        european_option = ql.VanillaOption(payoff, exercise)
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.stock_price))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.calculation_date, self.risk_free_rate, day_count))
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(self.calculation_date, self.dividend_rate, day_count))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self.calculation_date, calendar, self.volatility, day_count))
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, 
                                                   dividend_yield, 
                                                   flat_ts, 
                                                   flat_vol_ts)
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        bs_price = european_option.NPV()
        self.bs_price = bs_price
        return self.bs_price
    
    def data_set(self):
        '''
        Funtion to return a set of required data for one sample for training purpose.
        
        '''
        if self.bs_price == -1:
            self.BSM_price()
        maturity_in_year = (self.maturity - self.calculation_date)/365
        data_set = (self.stock_price, self.strike_price, maturity_in_year, self.dividend_rate, self.volatility, self.risk_free_rate, self.bs_price)
        return data_set

import datetime
import random
import pandas as pd

'''Date helper functions'''
def xldate_to_datetime(xldate):
    temp = datetime.datetime(1899, 12, 30)
    delta = datetime.timedelta(days=xldate)
    return temp+delta

def ql_to_datetime(d):
    return datetime.datetime(d.year(), d.month(), d.dayOfMonth())

def datetime_to_xldate(date):
    temp = datetime.datetime(1899, 12, 30)
    return (date - temp).days

def random_options(numbers = 0):
    options = []
    start_maturity = datetime.datetime(2020,10,30)
    end_maturity = datetime.datetime(2022,10,30)

    xldate1 = datetime_to_xldate(start_maturity)
    xldate2 = datetime_to_xldate(end_maturity)
    for number in range(numbers):
        maturity = ql.Date(random.randint(xldate1, xldate2+1))
        stock_price = random.randint(100, 501)
        strike_price = random.randint(7, 651)
        volatility = random.uniform(0.05, 0.90)
        dividend_rate = random.uniform(0.001, 0.003)
        risk_free_rate = random.uniform(0.001, 0.003)
        option_type = ql.Option.Call
        option = Option(calculation_date, maturity, stock_price, strike_price, volatility, dividend_rate, risk_free_rate, option_type)
        options.append(option.BSM_price())
    return options

def random_options_pd(numbers = 0):
    options = []
    start_maturity = datetime.datetime(2020,11,1)
    end_maturity = datetime.datetime(2023,10,30)
    calculation_date = datetime.datetime(2020,10,30)
    
    xldate1 = datetime_to_xldate(start_maturity)
    xldate2 = datetime_to_xldate(end_maturity)
    calculation_xldate = datetime_to_xldate(calculation_date)
    calculation_date = ql.Date(calculation_xldate)
    for number in range(numbers):
        maturity = ql.Date(random.randint(xldate1, xldate2+1))
        stock_price = random.randint(100, 501)
        strike_price = random.randint(7, 651)
        volatility = random.uniform(0.05, 0.90)
        dividend_rate = random.uniform(0, 0.003)
        risk_free_rate = random.uniform(0.001, 0.003)
        option_type = ql.Option.Call
        option = Option(calculation_date, maturity, stock_price, strike_price, volatility, dividend_rate, risk_free_rate, option_type)
        options.append(option.data_set())  
    dataframe = pd.DataFrame(options)
    dataframe.columns = ['stock_price', 'strike_price', 'maturity', 'devidends', 'volatility', 'risk_free_rate', 'call_price']
    return dataframe

training_data = random_options_pd(100000)
training_data.to_pickle('sample.pkl')


def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    n = 100000
    df = pd.read_pickle('sample.pkl')
    ## Normalize the data exploiting the fact that the BS Model is linear homogenous in S,K
    df['stock_price'] = df['stock_price']/df['strike_price']
    df['call_price'] = df['call_price']/df['strike_price']
    n_train =  (int)(0.8 * n)
    train = df[0:n_train]
    test = df[n_train+1:n]
    X_train = train[['stock_price', 'strike_price', 'maturity', 'devidends', 'volatility', 'risk_free_rate']].values
    y_train = train['call_price'].values
    X_test = test[['stock_price', 'strike_price', 'maturity', 'devidends', 'volatility', 'risk_free_rate']].values
    y_test = test['call_price'].values
    return X_train, y_train, X_test, y_test

def model(X_train, y_train, X_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model_mlp = Sequential()
    model_mlp.add(Dense({{choice([64,126, 256])}},
                        activation='relu', input_shape= (6,)))
    model_mlp.add(Dropout({{uniform(0, .3)}}))
    model_mlp.add(Dense({{choice([64, 126, 256])}}))
    model_mlp.add(Activation({{choice(['relu', 'elu'])}}))
    model_mlp.add(Dropout({{uniform(0, .3)}}))
    model_mlp.add(Dense({{choice([64, 126, 256])}}))
    model_mlp.add(Activation({{choice(['relu', 'elu'])}}))
    model_mlp.add(Dropout({{uniform(0, .3)}}))
    model_mlp.add(Dense({{choice([64, 126, 256])}}))
    model_mlp.add(Activation({{choice(['relu', 'elu'])}}))
    model_mlp.add(Dropout({{uniform(0, .3)}}))
    model_mlp.add(Dense(1))
    model_mlp.add(Activation({{choice(['softmax','linear'])}}))
    model_mlp.compile(loss='mean_squared_error', metrics=['mse'],
                  optimizer={{choice(['rmsprop', 'adam'])}})

    callbacks = [
    EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=4,
        verbose=1)
    ]
    
    model_mlp.fit( X_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=30,
              callbacks=callbacks,
              verbose=2,
              validation_split=0.2,
            shuffle=True)
    score, acc = model_mlp.evaluate(X_test, y_test, verbose=0)
    print('Test MSE:', acc)
    return {'loss': acc, 'status': STATUS_OK, 'model': model_mlp}


# In[2]:


if __name__ == '__main__':
    import gc; gc.collect()
    import tensorflow.python.keras.backend as K

    with K.get_session(): ## TF session
        trials=Trials()
        best_run, best_model = optim.minimize(model=model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=30,
                                              trials=trials)
        X_train, Y_train, X_test, Y_test = data()
        print("Evalutation of best performing model:")
        ## Normalize the data exploiting the fact that the BS Model is linear homogenous in S,K
        print(best_model.evaluate(X_test, Y_test))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)
        model = getBestModelfromTrials(trials)
        model.save_weights('model_weights.h5')
        print("Saved model weights to disk")


# In[ ]:




