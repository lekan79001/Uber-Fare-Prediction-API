
import streamlit as st
import joblib
import pandas as pd
import io

# Sidebar with instructions
st.sidebar.title('Instructions')
st.sidebar.markdown('''
1. Prepare your input data as a CSV file with the following columns:
   - VendorID, tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count, trip_distance, pickup_longitude, pickup_latitude, RatecodeID, store_and_fwd_flag, dropoff_longitude, dropoff_latitude, payment_type, fare_amount, extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge, total_amount
2. You can download a sample input below.
3. Upload your CSV file using the uploader.
4. View predictions and download results.
''')

# Show sample input
sample_data = '''VendorID,tpep_pickup_datetime,tpep_dropoff_datetime,passenger_count,trip_distance,pickup_longitude,pickup_latitude,RatecodeID,store_and_fwd_flag,dropoff_longitude,dropoff_latitude,payment_type,fare_amount,extra,mta_tax,tip_amount,tolls_amount,improvement_surcharge,total_amount\n1,2016-03-01 00:00:00,2016-03-01 00:07:55,1,2.5,-73.97674560546875,40.765151977539055,1,N,-74.00426483154298,40.74612808227539,1,9.0,0.5,0.5,2.05,0.0,0.3,12.35'''
st.sidebar.download_button('Download Sample CSV', sample_data, file_name='sample_input.csv', mime='text/csv')

st.title('🚕 Uber Data XGBoost Model Prediction')
st.markdown('Upload your CSV file below to get predictions from the trained model.')

# Load the trained XGBoost model
try:
    model = joblib.load('xgb_model.joblib')
except Exception as e:
    st.error(f'Error loading model: {e}')
    st.stop()

uploaded_file = st.file_uploader('Upload a CSV file with features for prediction', type=['csv'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader('Input Data')
        st.dataframe(data)
        # Engineer features to match model training
        data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
        data['pickup_hour'] = data['tpep_pickup_datetime'].dt.hour
        data['is_rush_hour'] = data['pickup_hour'].apply(
            lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
        )
        data['pickup_in_manhattan'] = (
        (data['pickup_latitude'].between(40.70, 40.83)) &
        (data['pickup_longitude'].between(-74.02, -73.93))
        ).astype(int)
        data['distance_rush_interaction'] = (
        data['trip_distance'] * data['is_rush_hour']
        )

        # Select only model features
        features = ['trip_distance', 'pickup_hour', 'is_rush_hour',
                    'pickup_in_manhattan', 'distance_rush_interaction']
        data = data[features]

        # Predict using the loaded model
        predictions = model.predict(data)
        predictions = predictions.clip(min=0)
        pred_df = pd.DataFrame(predictions, columns=['Prediction'])
        st.subheader('Predictions')
        st.dataframe(pred_df)

        # Download predictions
        csv = pred_df.to_csv(index=False)
        st.download_button('Download Predictions as CSV', csv, file_name='predictions.csv', mime='text/csv')
    except Exception as e:
        st.error(f'Error processing file or making predictions: {e}')
else:
    st.info('Please upload a CSV file to get predictions.')
