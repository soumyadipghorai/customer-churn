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

PhoneServiceEncoder = pickle.load(open('data/ML_models/PhoneServiceEncoder.pkl', 'rb'))
OnlineSecurityEncoder = pickle.load(open('data/ML_models/OnlineSecurityEncoder.pkl', 'rb'))
ContractEncoder = pickle.load(open('data/ML_models/ContractEncoder.pkl', 'rb'))
PaperlessBillingEncoder = pickle.load(open('data/ML_models/PaperlessBillingEncoder.pkl', 'rb'))

model = pickle.load(open('data/ML_models/stackedModel.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index() : 
    value = 'Predict your true worth using'
    PhoneService = list(df.PhoneService.unique())
    OnlineSecurity = list(df.OnlineSecurity.unique())
    Contract = list(df.Contract.unique())
    PaperlessBilling = list(df.PaperlessBilling.unique())

    PhoneService.sort()
    OnlineSecurity.sort()
    Contract.sort()
    PaperlessBilling.sort()

    return render_template(
        'index.html', 
        PhoneServiceList = PhoneService, 
        OnlineSecurityList = OnlineSecurity, 
        ContractList = Contract, 
        PaperlessBillingList = PaperlessBilling
    )

@app.route('/predict', methods = ['POST'])
def predict() : 

    if request.method == 'POST' : 
        tenure = float(request.form['tenure'])
        PhoneService = request.form['PhoneService']
        OnlineSecurity = request.form['OnlineSecurity']
        Contract = request.form['Contract']
        PaperlessBilling = request.form['PaperlessBilling']
        MonthlyCharges = float(request.form['MonthlyCharges'])

      
        PhoneServiceEncoded = PhoneServiceEncoder.transform([str(PhoneService)])[0]
        OnlineSecurityEncoded  = OnlineSecurityEncoder.transform([str(OnlineSecurity)])[0]
        ContractEncoded  = ContractEncoder.transform([str(Contract)])[0]
        PaperlessBillingEncoded  = PaperlessBillingEncoder.transform([str(PaperlessBilling)])[0]

        logging.info('Encoding Done : ')
        logging.debug('PhoneService -->'+ str(PhoneService)) 
        logging.debug('OnlineSecurity -->' + str(OnlineSecurity))
        logging.debug('Contract -->' + str(Contract))
        logging.debug('PaperlessBilling -->' + str(PaperlessBilling))

        to_predict = scaler.transform([[
            tenure, PhoneServiceEncoded, OnlineSecurityEncoded, ContractEncoded, PaperlessBillingEncoded, MonthlyCharges
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