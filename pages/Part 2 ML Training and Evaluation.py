# Import necessary libraries
import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy
import numpy as np
import joblib
import pickle

st.title("Part 2 : ML Training and Evaluation")

# Initialize st.session_state if not already initialized
if 'data' not in st.session_state:
    st.session_state.data = {}

# Retrieve the variables from st.session_state
x_train = st.session_state.data.get('x_train', None)
x_test = st.session_state.data.get('x_test', None)
y_train = st.session_state.data.get('y_train', None)
y_test = st.session_state.data.get('y_test', None)
ud = st.session_state.data.get('ud', None)

# Radio button to choose the model
model_choice = st.radio("Select a model:", ["SVM", "KNN", "Decision Tree", "Naive Bayes", "Logistic Regression", "Label Propagation"])

# Self training
x_train_model = copy.deepcopy(x_train)
ud_model = copy.deepcopy(ud)
y_train_model = copy.deepcopy(y_train)
x_test_model = copy.deepcopy(x_test)
y_test_model = copy.deepcopy(y_test)

# Display information about the data
if x_train_model is not None:
    st.write("x_train shape:", x_train_model.shape)
else:
    st.write("x_train_model is None.")
    
if y_train_model is not None:
    st.write("y_train shape:", y_train_model.shape)
else:
    st.write("y_train_model is None.")

if ud_model is not None:
    st.write("ud shape:", ud_model.shape)
else:
    st.write("ud_model is None.")

# button to train the model
train_button = st.button("Train Model")


# Initialize model
if model_choice == "SVM":
    model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, dual=False), method='sigmoid')
elif model_choice == "KNN":
    model = KNeighborsClassifier()
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
elif model_choice == "Naive Bayes":
    model = GaussianNB()
elif model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Label Propagation":
    model = LabelPropagation(kernel='knn', n_neighbors=5, gamma=30, max_iter=2000)

if train_button:
    max_iterations = 20

    progress_bar = st.progress(0)
    st.text("Training is in progress...")

    for iteration in range(max_iterations):
        progress_bar.progress((iteration + 1) / max_iterations)

        if model_choice == "Label Propagation":
            pseudo_labels = np.full(len(ud_model), -1)
            combined_X = np.concatenate((x_train_model, ud_model), axis=0)
            combined_y = np.concatenate((y_train_model, pseudo_labels), axis=0)
            model.fit(combined_X, combined_y)
            model_pred_unlabeled = model.predict(ud_model)
        else:
            model.fit(x_train_model, y_train_model)
            model_pred_unlabeled = model.predict(ud_model)

        confidence_threshold = 0.5
        confidence_model = model.predict_proba(ud_model)[:, 1] >= confidence_threshold

        y_train_model = pd.concat([y_train_model, pd.Series(model_pred_unlabeled[confidence_model])], ignore_index=True)
        x_train_model = pd.concat([x_train_model, ud_model[confidence_model]])

        # Using boolean indexing to drop rows
        ud_model = ud_model[~confidence_model].reset_index(drop=True)

    st.session_state.data['model'] = model
    st.session_state.data['model_trained'] = True
    st.success(f"Training finished for {model_choice}!")

model.fit(x_train_model, y_train_model)
st.session_state.data['model'] = model

# Evaluation
final_pred_model = model.predict(x_test_model)
accuracy_model = accuracy_score(y_test_model, final_pred_model)
score_model = f1_score(y_test_model, final_pred_model, average='weighted')
precision_model = precision_score(y_test_model, final_pred_model, average='weighted')
recall_model = recall_score(y_test_model, final_pred_model, average='weighted')

st.title("Model Evaluation")

if model_choice == "SVM":
    st.write(f"Final Accuracy of SVM: {accuracy_model}")
    st.write(f"Final F1 Score of SVM: {score_model}")
    st.write(f"Precision of SVM: {precision_model:.4f}")
    st.write(f"Recall of SVM: {recall_model:.4f}")
elif model_choice == "KNN":
    st.write(f"Final Accuracy of KNN: {accuracy_model}")
    st.write(f"Final F1 Score of KNN: {score_model}")
    st.write(f"Precision of KNN: {precision_model:.4f}")
    st.write(f"Recall of KNN: {recall_model:.4f}")
elif model_choice == "Decision Tree":
    st.write(f"Final Accuracy of Decision Tree: {accuracy_model}")
    st.write(f"Final F1 Score of Decision Tree: {score_model}")
    st.write(f"Precision of Decision Tree: {precision_model:.4f}")
    st.write(f"Recall of Decision Tree: {recall_model:.4f}")
elif model_choice == "Naive Bayes":
    st.write(f"Final Accuracy of Naive Bayes: {accuracy_model}")
    st.write(f"Final F1 Score of Naive Bayes: {score_model}")
    st.write(f"Precision of Naive Bayes: {precision_model:.4f}")
    st.write(f"Recall of Naive Bayes: {recall_model:.4f}")
elif model_choice == "Logistic Regression":
    st.write(f"Final Accuracy of Logistic Regression: {accuracy_model}")
    st.write(f"Final F1 Score of Logistic Regression: {score_model}")
    st.write(f"Precision of Logistic Regression: {precision_model:.4f}")
    st.write(f"Recall of Logistic Regression: {recall_model:.4f}")
elif model_choice == "Label Propagation":
    st.write(f"Final Accuracy of Label Propagation: {accuracy_model}")
    st.write(f"Final F1 Score of Label Propagation: {score_model}")
    st.write(f"Precision of Label Propagation: {precision_model:.4f}")
    st.write(f"Recall of Label Propagation: {recall_model:.4f}")


# Load the saved scaler, encoder, and model
loaded_scaler = joblib.load('scaler.pkl')
loaded_ordinal_encoder = joblib.load('encoder.pkl')
model = st.session_state.data.get('model', None)

def predict_delay(month, day_of_month, day_of_week, dep_time, unique_carrier, origin, dest, distance):
    try:
        # Create a DataFrame for the new data
        new_data = pd.DataFrame({
            'Month': [month],
            'DayOfMonth': [day_of_month],
            'DayOfWeek': [day_of_week],
            'DepTime': [dep_time],
            'UniqueCarrier': [unique_carrier],
            'Origin': [origin],
            'Dest': [dest],
            'Distance': [distance]
        })

        # Encode categorical features
        categorical_columns = ['UniqueCarrier', 'Origin', 'Dest']
        new_data[categorical_columns] = loaded_ordinal_encoder.transform(new_data[categorical_columns])

        # Scale all features
        for col in new_data.columns:
            new_data[col] = loaded_scaler.transform(new_data[col].values.reshape(-1, 1))

        # Prepare the input for prediction
        prediction_input = new_data.values.reshape(1, -1)

        # Make the prediction using the loaded model
        delay_prediction = model.predict(prediction_input)[0]

        return delay_prediction

    except KeyError as e:
        print(f"Error: {e}")
        return None

unique_carriers_list = ['AA', 'US', 'XE', 'OO', 'WN', 'NW', 'DL', 'OH', 'AS', 'UA', 'MQ', 'CO', 'EV', 'DH', 'YV', 'F9', 'AQ', 'TZ', 'HP', 'B6', 'FL', 'HA']
origin_airports_list = ['ATL', 'PIT', 'RDU', 'DEN', 'MDW', 'MEM', 'PBI', 'MSP', 'ONT', 'BDL', 'PHX', 'LAS',
    'DFW', 'DSM', 'CMH', 'ORF', 'SLC', 'CLT', 'GSO', 'IAD', 'SMF', 'FLL', 'DAL', 'ORD',
    'ITO', 'SAN', 'ROA', 'LGA', 'SFO', 'GSP', 'SEA', 'DAB', 'SJC', 'LIT', 'LAX', 'OAK',
    'COS', 'OKC', 'GRR', 'JFK', 'BOI', 'MCI', 'BWI', 'BHM', 'CRP', 'BOS', 'SAT', 'PHL',
    'STL', 'CIC', 'AUS', 'IAH', 'COD', 'HNL', 'RNO', 'BNA', 'TPA', 'MIA', 'EVV', 'PNS',
    'EWR', 'RSW', 'ANC', 'SNA', 'AMA', 'CID', 'DTW', 'DCA', 'LGB', 'MAF', 'MFE', 'BMI',
    'PDX', 'IPL', 'GRB', 'FAR', 'HOU', 'MTJ', 'DRO', 'MLU', 'VPS', 'TUL', 'CVG', 'SBA',
    'PWM', 'IDA', 'MCO', 'ACV', 'CHS', 'BGM', 'MSY', 'OGG', 'CLE', 'MOB', 'CAK', 'FAY',
    'SHV', 'TUS', 'IND', 'CAE', 'PVD', 'ROC', 'MFR', 'VLD', 'ELP', 'RIC', 'MKE', 'SGF',
    'TYS', 'CHO', 'EGE', 'BIS', 'JAN', 'JAX', 'BUF', 'MSO', 'BGR', 'CEC', 'ICT', 'MYR',
    'ALB', 'LIH', 'SBP', 'AEX', 'GNV', 'SAV', 'BTM', 'BRO', 'SJU', 'XNA', 'CPR', 'SDF',
    'JAC', 'AVL', 'PHF', 'GPT', 'SYR', 'PSP', 'MHT', 'MRY', 'CLD', 'FAT', 'MSN', 'ISP',
    'BUR', 'PSC', 'MEI', 'LEX', 'LBB', 'GEG', 'LFT', 'OMA', 'ISO', 'MGM', 'GRK', 'AVP',
    'ABQ', 'SRQ', 'BTV', 'FLG', 'BTR', 'MDT', 'ABI', 'TRI', 'ADQ', 'FSM', 'SMX', 'RST',
    'RAP', 'ILM', 'SIT', 'EKO', 'DBQ', 'CHA', 'BQK', 'BZN', 'MOD', 'MOT', 'MLB', 'TVC',
    'LAN', 'DAY', 'HSV', 'EUG', 'SGU', 'ACT', 'AGS', 'CLL', 'HLN', 'LNK', 'ASE', 'HRL',
    'ATW', 'CMI', 'LWS', 'DHN', 'FNT', 'FLO', 'RDM', 'TYR', 'KOA', 'FAI', 'OME', 'RDD',
    'MCN', 'TLH', 'MQT', 'AZO', 'FCA', 'CRW', 'TOL', 'HPN', 'FSD', 'FWA', 'SUN', 'LAW',
    'YUM', 'PIA', 'GTF', 'ACY', 'PIH', 'SPS', 'MLI', 'BIL', 'TWF', 'HTS', 'SBN', 'PFN',
    'GJT', 'CSG', 'JNU', 'TXK', 'LRD', 'BQN', 'CWA', 'SWF', 'GTR', 'BFL', 'OXR', 'KTN',
    'PIE', 'SCE', 'PSG', 'DLH', 'SJT', 'GUC', 'SPI', 'IYK', 'ABY', 'STT', 'ABE', 'GFK',
    'HDN', 'CDV', 'MBS', 'TUP', 'LCH', 'EYW', 'OTZ', 'ADK', 'GGG', 'VIS', 'GST', 'LYH',
    'HVN', 'BRW', 'LSE', 'ERI', 'HKY', 'BET', 'CDC', 'OAJ', 'WRG', 'ACK', 'DLG', 'YAK',
    'AKN', 'TEX', 'STX', 'SCC', 'APF', 'BPT', 'WYS', 'RFD', 'BLI', 'ILG', 'VCT', 'LWB',
    'PSE']
dest_airports_list = ['DFW', 'MCO', 'CLE', 'MEM', 'OMA', 'LGA', 'CVG', 'PSC', 'STL', 'SEA', 'SJC', 'ORD',
    'IAH', 'ATL', 'CMH', 'ILM', 'MSP', 'PHX', 'AUS', 'SYR', 'LAX', 'HNL', 'ORF', 'MYR',
    'PDX', 'CHS', 'SMF', 'DTW', 'SNA', 'PHL', 'IAD', 'LAS', 'OGG', 'ABI', 'SLC', 'EWR',
    'PVD', 'SFO', 'RDU', 'MDW', 'DAL', 'SAT', 'RNO', 'JAX', 'HRL', 'JAN', 'ISP', 'TPA',
    'JFK', 'FCA', 'PIT', 'CLT', 'BUF', 'OKC', 'ANC', 'SAN', 'BET', 'FLL', 'SBA', 'GJT',
    'DEN', 'MIA', 'BZN', 'FWA', 'BDL', 'BOI', 'GSO', 'GSP', 'OAK', 'LAN', 'YUM', 'MKE',
    'MQT', 'ONT', 'ELP', 'LBB', 'PBI', 'SJU', 'ACY', 'CAK', 'ABQ', 'KOA', 'BNA', 'IDA',
    'EVV', 'LEX', 'ITO', 'SBP', 'BOS', 'KTN', 'SGF', 'HOU', 'ALB', 'TUL', 'ABE', 'AMA',
    'LIT', 'IND', 'ROA', 'ROC', 'SAV', 'TOL', 'FAT', 'MCI', 'MSN', 'DCA', 'GRR', 'BUR',
    'AVL', 'MLU', 'ACV', 'RIC', 'COS', 'DAY', 'GRB', 'CRP', 'FNT', 'ICT', 'TLH', 'XNA',
    'CRW', 'GEG', 'HPN', 'VPS', 'CMI', 'LNK', 'EYW', 'BWI', 'MAF', 'HSV', 'TUS', 'PSP',
    'SHV', 'MSY', 'LIH', 'FAR', 'EUG', 'SDF', 'MHT', 'DSM', 'FLO', 'ISO', 'MFE', 'RDD',
    'BHM', 'BMI', 'FSD', 'RSW', 'FAY', 'CAE', 'MTJ', 'LWB', 'CLD', 'RAP', 'PWM', 'GPT',
    'TYS', 'SRQ', 'MBS', 'AVP', 'AZO', 'CEC', 'MFR', 'DAB', 'TUP', 'CID', 'PNS', 'JNU',
    'GNV', 'MRY', 'SGU', 'MOB', 'PHF', 'LAW', 'MEI', 'LFT', 'EGE', 'GUC', 'TRI', 'LGB',
    'BIL', 'CSG', 'SCE', 'GTF', 'BFL', 'FAI', 'EKO', 'DRO', 'BQK', 'TVC', 'MDT', 'CLL',
    'HTS', 'LSE', 'MSO', 'JAC', 'RDM', 'MLI', 'MOD', 'CPR', 'VLD', 'BTR', 'CHA', 'STT',
    'MOT', 'BTV', 'SPS', 'PIH', 'PIA', 'LWS', 'SMX', 'BGR', 'SJT', 'STX', 'VIS', 'HDN',
    'ACT', 'GRK', 'BRO', 'FSM', 'LCH', 'DBQ', 'DHN', 'MLB', 'IPL', 'PFN', 'GGG', 'SBN',
    'OXR', 'IYK', 'LRD', 'TWF', 'PSG', 'RST', 'BQN', 'BPT', 'AEX', 'ERI', 'CIC', 'MGM',
    'CDC', 'PIE', 'BIS', 'SWF', 'DLG', 'SUN', 'COD', 'ASE', 'ATW', 'OME', 'GTR', 'CHO',
    'GFK', 'BLI', 'ABY', 'TXK', 'APF', 'DLH', 'TYR', 'LYH', 'SPI', 'HLN', 'YAK', 'CWA',
    'MCN', 'ILG', 'AKN', 'SIT', 'FLG', 'HVN', 'BRW', 'BTM', 'BGM', 'OTZ', 'SCC', 'CDV',
    'WRG', 'AGS', 'HKY', 'RFD', 'SOP', 'WYS', 'PSE', 'ADQ', 'GST', 'TEX', 'ACK', 'TTN',
    'VCT']

# Streamlit UI
st.title("Flight Delay Prediction")

# Input form
month = st.slider("Select Month", 1, 12, 1)
day_of_month = st.slider("Select Day of Month", 1, 31, 1)
day_of_week = st.slider("Select Day of Week", 1, 7, 1)
dep_time = st.slider("Select Departure Time", 0, 2400, 1200)
unique_carrier = st.selectbox("Select Unique Carrier", unique_carriers_list)
origin = st.selectbox("Select Origin Airport", origin_airports_list)
dest = st.selectbox("Select Destination Airport", dest_airports_list)
distance = st.slider("Select Distance", 0, 5000, 1000)

# Predict button
if st.button("Predict Delay"):
    delay_prediction = predict_delay(month, day_of_month, day_of_week, dep_time, unique_carrier, origin, dest, distance)
    if delay_prediction is not None:
        if delay_prediction == 0:
            st.success("No Delay Expected.")
        elif delay_prediction == 1:
            st.warning("Possible Delay of 15 minutes or more.")
        else:
            st.error("Invalid Prediction Result.")
    else:
        st.error("Error in prediction. Please check your input.")