 
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler,QuantileTransformer
# loading the trained model
pickle_in = open('XGBclassifier_96.pkl', 'rb') 
classifier = pickle.load(pickle_in)
qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')

@st.cache()
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Pregnancies, Glucose, BP, ST, Insulin, BMI, DPF, Age):   
    #df = pd.DataFrame({'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BP': BP,'ST':ST,
                      #'Insulin':Insulin,'BMI':BMI,'DPF':DPF,'Age':Age} ,index=[0]) 
    # Pre-processing user input 
    dt = pd.read_csv('full_dataset.csv')
    dt = dt.drop('Outcome',axis=1)
    dt.loc[len(dt)] = [Pregnancies, Glucose, BP, ST, Insulin, BMI, DPF, Age]
    cols = ['Glucose','DPF', 'BMI']
    transformed_data = qt.fit_transform(dt[cols])
    dt[cols]= transformed_data
    dt['Age'] = np.log(dt['Age'])
    stand = StandardScaler().fit(dt.values)
    ds =np.array(dt.iloc[-1].tolist()).reshape(1,-1)
    data = stand.transform(ds)

    # Making predictions 
    prediction = classifier.predict(data)
    if prediction == 0:
        pred = 'Negative'
    else:
        pred = 'Positive'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Diabetes Prediction App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    # following lines create boxes in which user can enter data required to make prediction 
    Pregnancies = int(st.number_input('Pregnancies'))
    Glucose = st.number_input('Glucose Rate') 
    BP = st.number_input("Blood Pressure") 
    ST = st.number_input("SkinThickness")
    Insulin = st.number_input('Insulin Rate')
    BMI = st.number_input('BMI')
    DPF = st.number_input('DiabetesPedigreeFunction')
    Age = int(st.number_input('Age'))


    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Pregnancies, Glucose, BP, ST, Insulin, BMI, DPF, Age) 
        st.success('The case is {}'.format(result))
       # print(LoanAmount)
     
if __name__=='__main__': 
    main()