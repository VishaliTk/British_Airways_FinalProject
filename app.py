import streamlit as st
import numpy as np
import joblib

# ---------------- Load Model and Scaler ----------------
# Make sure lr_model.pkl and scaler.pkl are in the same folder
model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- App Title ----------------
st.set_page_config(page_title="British Airways Ticket Prediction", layout="centered")
st.title("✈️ British Airways Ticket Booking Prediction")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/British_Airways_Logo.svg/2560px-British_Airways_Logo.svg.png", width=400)
st.write("""
Predict whether a customer is likely to book a flight ticket based on their details.
""")

# ---------------- Input Fields ----------------
st.header("Enter Customer Details:")

num_passengers = st.number_input("Number of Passengers", 1, 10)
purchase_lead = st.number_input("Purchase Lead (days before flight)", 0, 365)
length_of_stay = st.number_input("Length of Stay (days)", 1, 30)
flight_hour = st.number_input("Flight Hour (0-23)", 0, 23)
flight_duration = st.number_input("Flight Duration (hours)", 1, 20)

wants_extra_baggage = st.selectbox("Extra Baggage", [0, 1])
wants_preferred_seat = st.selectbox("Preferred Seat", [0, 1])
wants_in_flight_meals = st.selectbox("In-flight Meals", [0, 1])

# ---------------- Predict Button ----------------
if st.button("Predict"):
    # Arrange input in correct order
    input_data = np.array([[num_passengers, purchase_lead, length_of_stay,
                            flight_hour, flight_duration,
                            wants_extra_baggage,
                            wants_preferred_seat,
                            wants_in_flight_meals]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]  # probability of booking

    # Display result
    if prediction[0] == 1:
        st.success(f"✅ Customer is likely to BOOK the ticket! (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Customer is NOT likely to book the ticket. (Probability: {probability:.2f})")

# ---------------- Footer ----------------
st.markdown("---")
st.write("Developed using Python, Streamlit & Logistic Regression")