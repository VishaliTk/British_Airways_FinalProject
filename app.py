import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("rf_model.pkl", "rb"))

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Home", "Prediction", "About", "Info"])

# ---------------- HOME PAGE ----------------
if menu == "Home":
    st.title("✈️ British Airways Customer Prediction System")

    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/British_Airways_Logo.svg/2560px-British_Airways_Logo.svg.png")

    st.write("""
    Welcome to the Airline Customer Behavior Prediction App.
    
    This application predicts whether a customer will book a ticket or not 
    using a Machine Learning model trained on airline data.
    """)

# ---------------- PREDICTION PAGE ----------------
elif menu == "Prediction":
    st.title("🔍 Predict Ticket Booking")

    num_passengers = st.number_input("Number of Passengers", 1, 10)
    purchase_lead = st.number_input("Purchase Lead", 0, 100)
    length_of_stay = st.number_input("Length of Stay", 1, 30)
    flight_hour = st.number_input("Flight Hour", 0, 23)
    flight_duration = st.number_input("Flight Duration", 1, 20)

    wants_extra_baggage = st.selectbox("Extra Baggage", [0,1])
    wants_preferred_seat = st.selectbox("Preferred Seat", [0,1])
    wants_in_flight_meals = st.selectbox("In-flight Meals", [0,1])

    if st.button("Predict"):
        input_data = np.array([[num_passengers, purchase_lead, length_of_stay,
                                flight_hour, flight_duration,
                                wants_extra_baggage,
                                wants_preferred_seat,
                                wants_in_flight_meals]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ Customer will BOOK the ticket")
        else:
            st.error("❌ Customer will NOT book the ticket")

# ---------------- ABOUT PAGE ----------------
elif menu == "About":
    st.title("ℹ️ About Project")

    st.write("""
    This project is based on customer behavior analysis for airline services.

    Techniques Used:
    - Web Scraping (BeautifulSoup, Requests)
    - NLP & Sentiment Analysis (NLTK, TextBlob)
    - Machine Learning (Random Forest, XGBoost)
    - Data Visualization (Matplotlib, Seaborn)

    The model predicts whether a customer will book a ticket based on various features.
    """)

# ---------------- INFO PAGE ----------------
elif menu == "Info":
    st.title("📊 Model Information")

    st.write("""
    Model Used: Random Forest Classifier  
    Accuracy: 91%  

    Key Features:
    - Purchase Lead
    - Flight Duration
    - Passenger Count
    - Extra Services (Meals, Baggage)

    This model helps airlines improve marketing strategies and customer experience.
    """)