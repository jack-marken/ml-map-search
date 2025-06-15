import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from model_lstm import LSTMModel
from model_gru import GRUModel
from model_rnn import RNNModel
from keras.models import Model
from keras.callbacks import EarlyStopping
from src.data.data import process_data
warnings.filterwarnings("ignore")

def train_model(model, X_train, y_train, name, config, scats):
    
    #X_train: ndarray(number, lags, scats), Input data for train.
    #y_train: ndarray(number, ), result data for train.
    #name: String, name of model.
    #config: Dict, parameter for train.
    #scats: gives the specific scats data to be used
    

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('model/'+ str(scats) + '/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + str(scats) + '/' + name + ' loss.csv', encoding='utf-8', index=False)
    

def model_trainer(argmodel, scats):
    lag = 4
    config = {"batch": 256, "epochs": 600}
    file1 = './src/data/SCATS_Data/'+str(scats)+'/'+str(scats)+'.csv'
    file2 = './src/data/SCATS_Data/'+str(scats)+'/'+str(scats)+'_test.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, lag) #Needs to be fixed before testing
    models = ['lstm','gru','rnn']
    if argmodel == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        lstm = LSTMModel([lag, 64, 64, 1])
        m = lstm.return_model()
        train_model(m, X_train, y_train, argmodel, config, scats)
    if argmodel == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        gru = GRUModel([lag, 64, 64, 1])
        m = gru.return_model()
        train_model(m, X_train, y_train, argmodel, config, scats)
    if argmodel == 'rnn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        rnn = RNNModel([lag, 64, 64, 1])
        m = rnn.return_model()
        train_model(m, X_train, y_train, argmodel, config, scats)

    if argmodel == 'all':
        for amodel in models:
            model_trainer(amodel,scats)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    parser.add_argument(
        "--scats",
        type=int,
        default=970,
        help="Specific Scat Value.")
    args = parser.parse_args()
    cnt = 0
    scatslist = [970,2000,2200,2820,2825,2827,2846,3001,3002,3120,3122,3126,3127,3180,3662,3682,3685,3804, 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321, 4324, 4335, 4812, 4821]
    if args.scats != 0:
        model_trainer(args.model, args.scats)
    else:
        for scatsite in scatslist:
            cnt+=1
            model_trainer(args.model, scatsite)

    models = ['lstm','gru','rnn']
    if args.model == "cheese":
        for model in models:
            for scatsite in scatslist:
                #cnt+=1
                model_trainer(args.model, scatsite)


if __name__ == '__main__':
    main(sys.argv)