import streamlit as st
import pickle
import numpy as np

#import the model
pipe = pickle.load(open('pipeline.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Price Predictor')

# brand
company = st.selectbox('Brand',df['Company'].unique())

#type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

#Ram
ram_type = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,32,64])

# weight
weight = st.number_input('Weight')

#touchscreen
touch = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screensize
screensize = st.number_input('Screen size')

# Resolution
resolution = st.selectbox(
    'Screen Resolution',
    [
        "1366x768",
        "1600x900",
        "1920x1080",
        "2304x1440",
        "2560x1440",
        "2560x1600",
        "2880x1800",
        "3200x1800",
        "3840x2160"
    ]
)

# Cpu
cpu = st.selectbox('Brand',df['Cpu brand'].unique())

# SSD
ssd = st.selectbox('SSD(in GB)',df['SSD'].unique())

# Gpu
gpu = st.selectbox('GPU',df['Gpu brand'].unique())

# os
os  = st.selectbox('OS',df['Operating System'].unique())

if st.button('Predict Price'):
    if touch == 'Yes':
        touch = 1
    else:
        touch = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    x_res = resolution.split('x')[0]
    y_res = resolution.split('x')[1]
    ppi = ((int(x_res)**2 + int(y_res)**2)**(1/2))/int(screensize)
    query = np.array([company,type,ram_type,weight,touch,ips,ppi,cpu,ssd,gpu,os])
    query = query.reshape(1,11)
    st.title("The Predicted Price of this Configuration is : " + str(int(np.exp(pipe.predict(query)[0]))))




