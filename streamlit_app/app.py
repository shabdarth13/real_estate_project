# streamlit_app/app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
from src.predict import predict_from_dict
import plotly.graph_objects as go

# -------------------------
# Page config and dark theme
# -------------------------
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")
st.markdown(
    """
    <style>
    /* FORCE DARK MODE EVERYWHERE */
    :root {
        --background-color: #0f111a !important;
        --text-color: #ffffff !important;
        --primary-color: #4a90e2 !important;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp  {
        background-color: #0f111a !important;
        color: #ffffff !important;
    }

    /* Input fields (number, text, selectbox, textarea) */
    input, textarea, select, .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox div, .stSelectbox select {
        background-color: #1a1c25 !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
    }

    /* Dropdown menu */
    .stSelectbox [role="listbox"] {
        background-color: #1a1c25 !important;
        color: #ffffff !important;
    }

    /* Form, expander, and containers */
    .stForm, .stExpander, .streamlit-expanderHeader {
        background-color: #14161f !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #333 !important;
    }

    /* Labels */
    label, .stMarkdown, .css-17eq0hr, .css-1offfwp, .stRadio label {
        color: #ffffff !important;
    }

    /* Subheaders and headers */
    h1, h2, h3, .stSubheader {
        color: #ffffff !important;
    }

    /* Buttons */
    button, .stButton button {
        background-color: #4a90e2 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: none !important;
    }
    button:hover {
        background-color: #357ab8 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.title("🏡 Real Estate Investment Advisor")
st.markdown("Fill all fields. Defaults are examples from your sample data.")

# -------------------------
# Property Form
# -------------------------
with st.form("property_form"):

    st.subheader("Basic Property Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        ID = st.number_input("ID", value=1, step=1)
        State = st.text_input("State", value="Tamil Nadu")
        City = st.text_input("City", value="Chennai")
        Locality = st.text_input("Locality", value="Locality_84")
        Property_Type = st.selectbox("Property_Type", ["Apartment","Independent House","Villa","Plot"], index=0)
        BHK = st.number_input("BHK", min_value=1, max_value=10, value=1)
    with col2:
        Size_in_SqFt = st.number_input("Size_in_SqFt", value=4740)
        Price_in_Lakhs = st.number_input("Price_in_Lakhs", value=489.76, format="%.2f")
        Price_per_SqFt = st.number_input("Price_per_SqFt", value=0.1, format="%.4f")
        Year_Built = st.number_input("Year_Built", min_value=1800, max_value=2025, value=1990)
        Furnished_Status = st.selectbox("Furnished_Status", ["Furnished","Unfurnished","Semi-furnished"], index=0)
    with col3:
        Floor_No = st.number_input("Floor_No", value=22)
        Total_Floors = st.number_input("Total_Floors", value=1)
        Age_of_Property = st.number_input("Age_of_Property", value=35)
        Nearby_Schools = st.number_input("Nearby_Schools", value=10)
        Nearby_Hospitals = st.number_input("Nearby_Hospitals", value=3)

    st.subheader("Additional Property Details")
    with st.expander("Advanced Options (Optional)"):
        col4, col5, col6 = st.columns(3)
        with col4:
            Public_Transport_Accessibility = st.selectbox("Public Transport", ["High","Medium","Low"], index=0)
            Parking_Space = st.selectbox("Parking_Space", ["Yes","No"], index=1)
        with col5:
            Security = st.selectbox("Security", ["Yes","No"], index=1)
            Facing = st.selectbox("Facing", ["East","West","North","South"], index=1)
        with col6:
            Owner_Type = st.selectbox("Owner_Type", ["Owner","Builder","Broker","Agent"], index=0)
            Availability_Status = st.selectbox("Availability_Status", ["Ready_to_Move","Under_Construction"], index=0)
            Amenities = st.text_input("Amenities", value="Playground, Gym, Garden, Pool, Clubhouse")

    submitted = st.form_submit_button("Predict Investment")

# -------------------------
# Prediction & Display
# -------------------------
if submitted:
    input_dict = {
        "ID": ID,
        "State": State,
        "City": City,
        "Locality": Locality,
        "Property_Type": Property_Type,
        "BHK": BHK,
        "Size_in_SqFt": Size_in_SqFt,
        "Price_in_Lakhs": Price_in_Lakhs,
        "Price_per_SqFt": Price_per_SqFt,
        "Year_Built": Year_Built,
        "Furnished_Status": Furnished_Status,
        "Floor_No": Floor_No,
        "Total_Floors": Total_Floors,
        "Age_of_Property": Age_of_Property,
        "Nearby_Schools": Nearby_Schools,
        "Nearby_Hospitals": Nearby_Hospitals,
        "Public_Transport_Accessibility": Public_Transport_Accessibility,
        "Parking_Space": Parking_Space,
        "Security": Security,
        "Amenities": Amenities,
        "Facing": Facing,
        "Owner_Type": Owner_Type,
        "Availability_Status": Availability_Status
    }

    result = predict_from_dict(input_dict)

    st.subheader("🏆 Prediction Result")
    if result["prediction"] == 1:
        st.success("✅ Good Investment")
    else:
        st.error("❌ Bad Investment")

    if result.get("probability"):
        st.subheader("Investment Probability")
        proba = result["probability"][0] 
        good_prob = round(proba[1]*100, 2)
        bad_prob = round(proba[0]*100, 2)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=good_prob,
            number={'suffix': "%", 'font': {'color': 'white', 'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            title={'text': "Good Investment Probability", 'font': {'color': 'white', 'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white"},
                'bar': {'color': "#0a47ff", 'thickness': 0.25},
                'bgcolor': "#0f111a",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': good_prob
                    }
                }
            ))
        fig.update_layout(
            paper_bgcolor="#0f111a",
            plot_bgcolor="#0f111a",
            )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Legend:**  
        - 🔴 Red zone: Low chance of good investment (0-50%)  
        - 🟡 Yellow zone: Moderate chance (50-75%)  
        - 🟢 Green zone: High chance (75-100%)  
        - ⚫ Needle points to your property's probability
        """)

    st.markdown("### Input Features Received by Model")
    st.json(result["input_features"])
