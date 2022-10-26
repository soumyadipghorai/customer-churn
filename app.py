import numpy as np
import pandas as pd 
from rich.logging import RichHandler
import pickle, logging
from sklearn.preprocessing import LabelEncoder
import xgboost
from flask import Flask, render_template, request 

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)  
logger = logging.getLogger("rich")

df = pd.read_csv('data/preprocessed_data.csv')
scaler = pickle.load(open('data/ML_models/StandardScaler.pkl', 'rb'))


# 'tenure',
#  'MonthlyCharges'

InternetServiceEncoder = pickle.load(open('data/ML_models/InternetServiceEncoder.pkl', 'rb'))
OnlineSecurityEncoder = pickle.load(open('data/ML_models/OnlineSecurityEncoder.pkl', 'rb'))
ContractEncoder = pickle.load(open('data/ML_models/ContractEncoder.pkl', 'rb'))
StreamingTVEncoder = pickle.load(open('data/ML_models/StreamingTVEncoder.pkl', 'rb'))
DependentsEncoder = pickle.load(open('data/ML_models/DependentsEncoder.pkl', 'rb'))

model = pickle.load(open('data/ML_models/stackedModel.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index() : 
    Dependents = list(df.Dependents.unique())
    InternetService = list(df.InternetService.unique())
    OnlineSecurity = list(df.OnlineSecurity.unique())
    Contract = list(df.Contract.unique())
    StreamingTV = list(df.StreamingTV.unique())

    Dependents.sort()
    InternetService.sort()
    OnlineSecurity.sort()
    Contract.sort()
    StreamingTV.sort()

    return render_template(
        'index.html', 
        DependentsList = Dependents, 
        InternetServiceList = InternetService, 
        OnlineSecurityList = OnlineSecurity, 
        ContractList = Contract, 
        StreamingTVList = StreamingTV
    )

@app.route('/predict', methods = ['POST'])
def predict() : 

    if request.method == 'POST' : 
        Dependents = request.form['Dependents']
        InternetService = request.form['InternetService']
        OnlineSecurity = request.form['OnlineSecurity']
        Contract = request.form['Contract']
        StreamingTV = request.form['StreamingTV']
        MonthlyCharges = float(request.form['MonthlyCharges'])

      
        DependentsEncoded = DependentsEncoder.transform([str(Dependents)])[0]
        InternetServiceEncoded = InternetServiceEncoder.transform([str(InternetService)])[0]
        OnlineSecurityEncoded  = OnlineSecurityEncoder.transform([str(OnlineSecurity)])[0]
        ContractEncoded  = ContractEncoder.transform([str(Contract)])[0]
        StreamingTVEncoded  = StreamingTVEncoder.transform([str(StreamingTV)])[0]

        logging.info('Encoding Done : ')
        logging.debug('Dependents -->'+ str(Dependents)) 
        logging.debug('InternetService -->'+ str(InternetService)) 
        logging.debug('OnlineSecurity -->' + str(OnlineSecurity))
        logging.debug('Contract -->' + str(Contract))
        logging.debug('StreamingTV -->' + str(StreamingTV))

        to_predict = scaler.transform([[
            DependentsEncoded, InternetServiceEncoded, OnlineSecurityEncoded, ContractEncoded, StreamingTVEncoded, MonthlyCharges
            ]])

        prediction = model.predict(to_predict)
        logging.debug("final prediction "+ str(prediction))

        if prediction == 1 :
            churn = ''
        else : 
            churn = 'NOT'

        if prediction < 0 : 
            return render_template('index.html', prediction_value = 'som value')  
        
        else : 
            return render_template(
                'prediction.html',  
                prediction_value = churn, 
            ) 
 
    else : 
        return render_template('index.html', prediction_value = "invalid response")
 
if __name__ == "__main__" :
    app.run(debug = True)