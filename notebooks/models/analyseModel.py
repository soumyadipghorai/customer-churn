import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc

def results(algorithm, x_train, y_train, x_test, y_test, of_type = None, columns = None, f_imp = True):
    
    print ("*"*30)
    print ("MODEL - OUTPUT")
    print ("*"*30)
    algorithm.fit(x_train,y_train)
    predictions = algorithm.predict(x_test)
    
    print ("\naccuracy_score :",accuracy_score(y_test, predictions))
    
    print ("\nclassification report :\n",(classification_report(y_test, predictions)))
        
    plt.figure(figsize=(14,12))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(y_test, predictions),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
    plt.title("CONFUSION MATRIX",fontsize=20)
    
    if of_type != None: 
        predicting_probabilites = algorithm.predict_proba(x_test)[:,1]
        fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
        plt.subplot(222)
        plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
        plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
        plt.legend(loc = "best")
        plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)
    
        if f_imp : 
            if  of_type == "feat":
                
                dataframe = pd.DataFrame(algorithm.feature_importances_, columns).reset_index()
                dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})
                dataframe = dataframe.sort_values(by="coefficients",ascending = False)
                
                plt.subplot(224)
                ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")
                plt.title("FEATURE IMPORTANCES",fontsize =20)
                for i,j in enumerate(dataframe["coefficients"]):
                    ax.text(.011,i,j,weight = "bold")
            
            elif of_type == "coef" :
                
                dataframe = pd.DataFrame(algorithm.coef_.ravel(),columns).reset_index()
                dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})
                dataframe = dataframe.sort_values(by="coefficients",ascending = False)  
                plt.subplot(224)
                ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")
                plt.title("FEATURE IMPORTANCES",fontsize =20)
                for i,j in enumerate(dataframe["coefficients"]):
                    ax.text(.011,i,j,weight = "bold")
                
        elif of_type == "none" :
            return (algorithm)

def generate_classification_report(model_name, x_train, y_train, x_test, y_test, of_type, columns) : 
    model_name.fit(x_train, y_train)
    y_pred = model_name.predict(x_test)
    print('='*25)
    print('avg cross validation score : ',round(np.mean(cross_val_score(model_name, x_train, y_train, cv=10)), 3))
    print('='*25)
    print('Accracy score : ', (round(accuracy_score(y_test, y_pred), 3))*100, '%')
    print('='*25)
    print('classification report : ')
    print(classification_report(y_test,y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    print("="*25)
    TN = cf_matrix[1][1]
    FP = cf_matrix[0][1]
    print("specificity : {}".format(TN/(TN + FP)))
    print('='*25)
    print('confusion matrix heatmap : \n')
    sns.heatmap(cf_matrix, annot=True)
    return y_pred