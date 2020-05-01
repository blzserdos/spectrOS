def SNV(data):
    '''
    Takes a pandas dataframe of NxM matrix (N-number of samples, M-wavelengths) and performes SNV spectral correction on the therein contained spectra. 
    '''
    
    import numpy as np
    import pandas as pd

    
    snv_spectra = (data.values-data.values.mean(axis=1).reshape(-1,1))/data.values.std(axis=1).reshape(-1,1)
    data_snv = pd.DataFrame(snv_spectra, index=data.index, columns=data.columns)
    return data_snv
    
def load_data(snv_corr=True):
    '''
    Loads and creates the measurement table raw, NIR, UV and full.
    full is a 1014 x 1337 dataframe, containing the UV (columns: 0:911), NIR (columns: 911:2013) and composition (columns: 2013:2021) measurements.
    When SNV = True, performs SNV spectral correction, SNV = False returns raw spectra.
    '''
    
    import numpy as np
    import pandas as pd
    from uv_nir_gos import SNV

    UV = pd.read_excel('~/gos/UV.xlsx', sheet_name=0, header=0)
    UV.index = UV['code'].values
    UV.drop(['code', 'sample number'], axis=1, inplace=True)
    UV.drop(['A+1', 'A+2', 'B+1', 'B+2'], axis=0, inplace=True)
    
    old_idx = UV.index.to_list()
    new_idx = old_idx

    for idx, i in enumerate(old_idx):
        if len(i) == 2:
            new_idx[idx] = i[0] + str(0) + i[1]
    UV.index = new_idx
    
    NIR = pd.read_csv('~/gos/raw.csv', index_col='Unnamed: 0')
    NIR.columns = np.linspace(780,2500,1102)
    NIR = NIR.iloc[:, (NIR.columns >= 1100) & (NIR.columns <=1501) | (NIR.columns >= 2200) & (NIR.columns <= 2451)] # select relevant wavelength ranges (based on Physics Dep.)
    
    if snv_corr == True:
        UV = SNV(UV)
        NIR = SNV(NIR)
    
    UVsubset = UV.iloc[:, (UV.columns >= 190) & (UV.columns <=240) | (UV.columns >= 900) & (UV.columns <= 1100)]
    
    comp = pd.read_csv('~/gos/comp.csv', index_col='Unnamed: 0')
    raw = pd.merge(NIR, comp,on=NIR.index);
        
    for i in range(raw.shape[0]):
        raw.loc[i, 'key_0'] = raw.loc[i,'key_0'].split('_')[2]
        
    raw.rename(str, columns={'key_0': 'sample'}, inplace=True);
    raw.index = raw['sample'].values
    raw.drop('sample', axis=1, inplace=True)
    #raw.drop('B01', axis=0, inplace=True) # outlier
    
    full = UV.merge(raw, left_on=UV.index, right_on=raw.index, how='inner')
    
    full.index = full['key_0'].values
    full.drop('key_0', axis=1, inplace=True)
    full.sort_values('Y_DP2', ascending=False, inplace=True)
    full_r = UVsubset.merge(raw, left_on=UVsubset.index, right_on=raw.index, how='inner')
    full_r.index = full_r['key_0'].values
    full_r.drop('key_0', axis=1, inplace=True)
    full_r.sort_values('Y_DP2', ascending=False, inplace=True)    

    
    return raw, NIR, UV, full_r, full

################################################################################

def train_test_split(full, test='rand'):
    
    '''
    Allocate train and test sets.
    By default, the dataframe full is in descending order by DP2.
    If test = 'rand', the test set comprises of random samples
    If test = '4th' the test set comprises of every 4th sample
    If test = 'E' the test set comprises of every sample in experiment 'B'
    '''
    import numpy as np
    import pandas as pd
    import random

    ids = full.index.to_list()
    if test == '4th':
        test_i = random.choices(list(set(ids)), k=42)
        train_i = list(set(ids)-set(test_i))
        Train = full.loc[train_i,:]
        Test = full.loc[test_i,:]
        
    elif test == 'rand':
        test_i = random.choices(list(set(ids)), k=42)
        train_i = list(set(ids)-set(test_i))
        Train = full.loc[train_i,:]
        Test = full.loc[test_i,:]
    
    elif test == 'E':
        mask = full.index.str.contains('E')
        test_i = full[mask].index.unique().to_list()
        train_i = list(set(ids)-set(test_i))
        Train = full.loc[train_i,:]
        Test = full.loc[test_i,:]        

    return test_i, Train, Test

##############################################################################

def wl_select(Train, Test, spec='ALLr'):
    if spec == 'UV': # all UV wavelengths
        X_train = Train.iloc[:,:911]
        X_test = Test.iloc[:,:911]
    if spec == 'UVr': # selected UV wavelengths
        ix = Train.iloc[:,:51].columns.to_list()
        ix.extend(Train.iloc[:,710:911].columns.to_list())
        ix.extend(Train.iloc[:,-8:].columns.to_list())
        X_train = Train[ix].iloc[:,:-8] 
        X_test = Test[ix].iloc[:,:-8]
    elif spec == 'NIR': # selected NIR wavelengths
        X_train = Train.iloc[:,911:1329]
        X_test = Test.iloc[:,911:1329]
    elif spec == 'ALL': # all UV and selected NIR wavelengths
        X_train = Train.iloc[:,:1329]
        X_test = Test.iloc[:,:1329]
    elif spec == 'ALLr': # selected UV and NIR wavelengths
        ix = Train.iloc[:,:51].columns.to_list()
        ix.extend(Train.iloc[:,710:].columns.to_list())
        X_train = Train[ix].iloc[:,:-8]
        X_test = Test[ix].iloc[:,:-8]
        
    Y_train = Train.iloc[:,-8:]
    Y_test = Test.iloc[:,-8:]
    return X_train, X_test, Y_train, Y_test

##############################################################################

def plsregress(Train, Test, devcomp=None, spec='ALLr'):
    '''
    Builds PLSR model using spectra data specified in spec. Plots error on the development set vs number of principle components.
    options: 'UV', UVr, 'NIR', 'ALLr', 'ALL'
    
    '''
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold, LeaveOneOut
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.utils import shuffle
    from uv_nir_gos import wl_select
    trainR2 = []
    devMSE = []
    devR2 = []

    X, _, Y, __ = wl_select(Train,Test,spec)
    
    X,Y = shuffle(X,Y)

    for i in np.arange(1, 20):    
        ytests = []
        ypreds = []
        train_score = []

        cv = LeaveOneOut() # higher error as expected compared to LOO -- unsure how Cao & co. got their results.
        sample_ids = list(set(X.index.tolist()))
        for train_idx, dev_idx in cv.split(sample_ids): 
            tr_ix = Train.iloc[train_idx,:].index.tolist()
            dev_ix = Train.iloc[dev_idx,:].index.tolist()

            X_train, X_dev = X.loc[tr_ix], X.loc[dev_ix]
            y_train, y_dev = Y.loc[tr_ix], Y.loc[dev_ix]
            
            # fit scaler to train apply to test
            scaler = MinMaxScaler()
            X_train_t = scaler.fit_transform(X_train.values)
            X_dev_t = scaler.transform(X_dev.values)

            pls2 = PLSRegression(n_components=i)
            pls2.fit(X_train_t, y_train.values)
            train_score.append(pls2.score(X_train_t, y_train.values))

            y_pred = pls2.predict(X_dev_t)

            ytests += list(y_dev.values)
            ypreds += list(y_pred)
        
          
        train_R2 = np.asarray(train_score).mean(axis=0)
        train_R2_std = np.asarray(train_score).std(axis=0)
        dev_R2 = r2_score(ytests, ypreds, multioutput='raw_values')
        dev_MSE = mean_squared_error(ytests, ypreds, multioutput='raw_values')

        devMSE.append(dev_MSE)
        devR2.append(dev_R2)
        trainR2.append(train_R2)
    if devcomp != None:    
      resDF = pd.DataFrame([np.asarray(devR2)[devcomp,:],np.asarray(devMSE)[devcomp,:]], columns=Y.columns,index=['R2', 'MSE'])
      resDF.to_csv('gos/results/resDF_'+str(spec)+'_PLSR_dev.csv')
    # Plot results
    plt.plot(np.arange(1, 20), np.array(devMSE), '-o');
    plt.xlabel('Number of principal components in regression')
    plt.ylabel('MSE')
    plt.legend(Y.columns.to_list());
    plt.xlim(left=0, right=21)
    plt.savefig('gos/results/PLSR_dev_' +spec+'.png')
    plt.close()

######################################################################

def PLSR_test(Train, Test, ncomp, spec='UV'):

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.utils import shuffle
    from uv_nir_gos import wl_select

    X_train, X_test, Y_train, Y_test = wl_select(Train,Test,spec)
    
    X_train, Y_train = shuffle(X_train, Y_train)

    scaler = MinMaxScaler()
    X_train_t = scaler.fit_transform(X_train.values)
    X_test_t = scaler.transform(X_test.values)
    
    pls2 = PLSRegression(n_components=ncomp)
    pls2.fit(X_train_t, Y_train.values)
    y_pred = pls2.predict(X_test_t)
    
    test_R2 = r2_score(Y_test.values, y_pred, multioutput='raw_values')
    test_MSE = mean_squared_error(Y_test.values, y_pred, multioutput='raw_values')
    
    plt.scatter(Y_test.values, y_pred);
    plt.xlabel('Measured w/w%')
    plt.ylabel('Predicted w/w%')
    #plt.legend(Y.columns.to_list());
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.savefig('gos/results/PLSR_test_' +spec+'.png')
    plt.close()

    resDF = pd.DataFrame([test_R2,test_MSE], columns=Y_test.columns,index=['R2', 'MSE'])
    resDF.to_csv('gos/results/resDF_'+str(spec)+'_PLSR_test.csv')
    print(resDF)
    return resDF

#####################################################################

def build_model(X_train, L1, L2, optim = 'rmsprop', model_type='ff'):
    
    from keras import models
    from keras import layers
    
    if model_type == 'ff':
      if L2 != None:
        model = models.Sequential()
        model.add(layers.Dense(L1, activation='relu', 
                               input_shape=(X_train.shape[1],))) # train_X
        model.add(layers.Dense(L2, activation='relu'))
        model.add(layers.Dense(8))
        model.compile(optimizer=optim, loss='mse', metrics=['mae'])
      elif L2 == None:
        model = models.Sequential()
        model.add(layers.Dense(L1, activation='relu', 
                               input_shape=(X_train.shape[1],))) # train_X
        model.add(layers.Dense(8))
        model.compile(optimizer=optim, loss='mse', metrics=['mae']) 
              
    elif model_type == '1dconv':
        model = models.Sequential()
        model.add(layers.Conv1D(20, 10, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(layers.AveragePooling1D())
        model.add(layers.Conv1D(12, 7, activation='relu'))
        model.add(layers.MaxPooling1D())
        model.add(layers.Conv1D(6, 5, activation='relu'))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(8, activation='linear'))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

######################################################################

def train_ann(Train, Test, layersizes, opti = 'rmsprop', spec='UV', modeltype='ff'):
    #optim = opti
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold, LeaveOneOut
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.utils import shuffle
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from uv_nir_gos import wl_select

    X, _, Y, __ = wl_select(Train,Test,spec)
    
    X,Y = shuffle(X,Y)

    validation_scores = []
    val_R2 = []
    val_MSE = []
    train_MSE_history = []
    val_MSE_history = []
    val_MAE_history = []
    ytests = []
    ypreds = []
    cv = LeaveOneOut() # higher error as expected compared to LOO -- unsure how Cao & co. got their results.
    num_epochs = 500
    
    i=1
    
    sample_ids = list(set(X.index.tolist()))
    for train_idx, dev_idx in cv.split(sample_ids): 
        tr_ix = Train.iloc[train_idx,:].index.tolist()
        dev_ix = Train.iloc[dev_idx,:].index.tolist()

        X_train, X_dev = X.loc[tr_ix].values, X.loc[dev_ix].values
        Y_train, Y_dev = Y.loc[tr_ix].values, Y.loc[dev_ix].values

        scaler = MinMaxScaler()
        X_train_t = scaler.fit_transform(X_train)
        X_dev_t = scaler.transform(X_dev)
        #X_train_t = np.expand_dims(X_train_t, axis=2) # for conv1d !!remove for DENSE network!!!!!
        #X_dev_t = np.expand_dims(X_dev_t, axis=2) # for conv1d
        model = None # Clearing the NN.
        model = build_model(X_train_t, layersizes[0], layersizes[1], optim=opti, model_type=modeltype)
        print('processing fold #', i)
        
        train_history = model.fit(X_train_t, Y_train, epochs=num_epochs, 
                                  validation_data=(X_dev_t, Y_dev), batch_size=1, verbose=0) # batch_size=1, 
        
        y_pred = model.predict(X_dev_t)

        ytests += list(Y_dev)
        ypreds += list(y_pred)    
        
        train_loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        val_mae = train_history.history['val_mean_absolute_error']
        
        train_MSE_history.append(train_loss) #append elements to the empty lists we created above
        val_MSE_history.append(val_loss)
        val_MAE_history.append(val_mae)
        validation_score = model.evaluate(X_dev_t, Y_dev, verbose=0)
        validation_scores.append(validation_score)    
        i += 1
        dev_R2 = r2_score(ytests, ypreds, multioutput='raw_values')
        dev_MSE = mean_squared_error(ytests, ypreds, multioutput='raw_values') 
        val_R2.append(dev_R2)
        val_MSE.append(dev_MSE)   
    validation_score = np.average(validation_scores)
    #ANN_MVP_NIR = pd.DataFrame(np.hstack((np.asarray(ytests), np.asarray(ypreds))))

    avg_train_mse_history = [np.mean([x[i] for x in train_MSE_history]) for i in range(num_epochs)]
    avg_val_mse_history = [np.mean([x[i] for x in val_MSE_history]) for i in range(num_epochs)]
    #avg_val_mae_history = [np.mean([x[i] for x in val_MAE_history]) for i in range(num_epochs)]
    plt.plot(range(1, len(avg_train_mse_history) + 1), avg_train_mse_history)
    plt.plot(range(1, len(avg_val_mse_history) + 1), avg_val_mse_history)

    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend(['train_MAE', 'val_MSE'])
    #plt.show()
    plt.savefig('gos/results/ANN_dev_' +spec+'.png')
    #plt.close()
    resDF = pd.DataFrame([np.asarray(val_R2).mean(axis=0),np.asarray(val_MSE).mean(axis=0)],columns=Y.columns,index=['R2', 'MSE'])
    resDF.to_csv('gos/results/resDF_'+str(spec)+'_ANN_dev.csv')
    print(resDF)
    return val_R2, val_MSE
#################################################################

def ANN_test(Train, Test, layersizes, spec='UV', modeltype='ff'):

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.utils import shuffle
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from uv_nir_gos import wl_select

    X_train, X_test, Y_train, Y_test = wl_select(Train,Test,spec)
    X_train, Y_train = shuffle(X_train, Y_train)

    scaler = MinMaxScaler()
    X_train_t = scaler.fit_transform(X_train.values)
    X_test_t = scaler.transform(X_test.values)
    
    model = build_model(X_train_t, layersizes[0], layersizes[1], model_type=modeltype)
    model.fit(X_train_t, Y_train, epochs=200, batch_size=1, verbose=0) #batch_size=1, 
    test_scores = model.evaluate(X_test_t, Y_test, verbose=0)
    ypreds = model.predict(X_test_t)
    
    test_R2 = r2_score(Y_test.values, ypreds, multioutput='raw_values')
    test_MSE = mean_squared_error(Y_test.values, ypreds, multioutput='raw_values')
    
    plt.scatter(Y_test.values, ypreds);
    plt.xlabel('Measured w/w%')
    plt.ylabel('Predicted w/w%')
    #plt.legend(Y.columns.to_list());
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.savefig('gos/results/ANN_test_' +spec+'.png')
    plt.close()
    
    resDF = pd.DataFrame([test_R2,test_MSE], columns=Y_test.columns,index=['R2', 'MSE'])
    resDF.to_csv('gos/results/resDF_'+str(spec)+'_ANN_test.csv')
    print(resDF)
    return resDF