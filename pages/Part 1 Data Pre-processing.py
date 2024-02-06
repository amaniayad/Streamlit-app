import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def handle_outliers(data):
    a = np.percentile(data['Distance'], 75)
    b = np.percentile(data['Distance'], 25)
    IQRA = a - b
    UpperA = (a + 1.5 * IQRA)
    LowerA = (b - 1.5 * IQRA)

    data['Distance'] = np.where(data['Distance'] > UpperA, UpperA, data['Distance'])
    data['Distance'] = np.where(data['Distance'] < LowerA, LowerA, data['Distance'])


def main():
    st.title("Part 1: Data Pre-processing")

    # Reading datasets
    df = pd.read_csv("./pages/labeled/labeled.csv")
    ud = pd.read_csv("./pages/unlabeled/unlabeled.csv")  

    # Display data
    st.subheader("Labeled Data:")
    st.dataframe(df)

    st.subheader("Unlabeled Data:")
    st.dataframe(ud)

    # Data statistics
    st.subheader("Data Statistics:")
    st.write(f"Labeled Data Shape: {df.shape}")
    st.write(f"Unlabeled Data Shape: {ud.shape}")
    st.write("\n")
    st.write("Labeled Data Info:")
    st.write(df.info())
    st.write("\n")
    st.write("Unlabeled Data Info:")
    st.write(ud.info())
    st.write("\n")
    st.write("Labeled Data Description:")
    st.write(df.describe())
    st.write("\n")
    st.write("Unlabeled Data Description:")
    st.write(ud.describe())

    # Data visualization
    st.subheader("Data Visualization:")
    st.write("Histogram of 'dep_delayed_15min':")
    st.plotly_chart(px.histogram(df, x='dep_delayed_15min'))
    st.write("Box plot of 'Distance' in Labeled Data:")
    st.plotly_chart(px.box(df, y='Distance'))
    st.write("Box plot of 'Distance' in Unlabeled Data:")
    st.plotly_chart(px.box(ud, y='Distance'))

    # Handling Outliers
    st.subheader("Handling Outliers:")
    handle_outliers(df)
    handle_outliers(ud)

    # Display box plots after handling outliers
    st.write("Box plot of 'Distance' in Labeled Data (after handling outliers):")
    st.plotly_chart(px.box(df, y='Distance'))
    st.write("Box plot of 'Distance' in Unlabeled Data (after handling outliers):")
    st.plotly_chart(px.box(ud, y='Distance'))

    # Gaussian Distribution
    st.subheader("Gaussian Distribution:")
    st.write("Gaussian Distribution of 'DepTime' in Labeled Data:")
    st.plotly_chart(px.histogram(df, x='DepTime', nbins=30))
    st.write("Gaussian Distribution of 'Distance' in Labeled Data:")
    st.plotly_chart(px.histogram(df, x='Distance', nbins=30))

# Encoding
    st.subheader("Encoding:")
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    categorical_columns = ['UniqueCarrier', 'Origin', 'Dest']
    df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])
    ud[categorical_columns] = ordinal_encoder.transform(ud[categorical_columns])
    joblib.dump(ordinal_encoder, 'ordinal_encoder.pkl')

    label_encoder = LabelEncoder()
    df['dep_delayed_15min'] = label_encoder.fit_transform(df['dep_delayed_15min'])
    joblib.dump(label_encoder, 'label_encoder.pkl')

# Display samples after encoding
    st.write("Sample of Labeled Data after Encoding:")
    st.dataframe(df.head())

    st.write("Sample of Unlabeled Data after Encoding:")
    st.dataframe(ud.head())

# Additional steps
    st.subheader("Additional Steps:")
    label_encoder = LabelEncoder()
    df['dep_delayed_15min'] = label_encoder.fit_transform(df['dep_delayed_15min'])

    x = df.drop(columns=['dep_delayed_15min'])
    y = df['dep_delayed_15min']

    cols = ['Month', 'DayofMonth', 'DayOfWeek']
    for i in range(len(cols)):
        df[cols[i]] = df[cols[i]].str.replace('c-', '').astype(float)
        x[cols[i]] = x[cols[i]].str.replace('c-', '').astype(float)
        ud[cols[i]] = ud[cols[i]].str.replace('c-', '').astype(float)

# Split the data
    st.subheader("Split the Data:")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

# Display split data
    st.write("Training Data (X_train):")
    st.dataframe(x_train.head())

    st.write("Testing Data (X_test):")
    st.dataframe(x_test.head())

    st.write("Training Labels (y_train):")
    st.dataframe(y_train.head())

    st.write("Testing Labels (y_test):")
    st.dataframe(y_test.head())


# Normalization
    st.subheader("Normalization:")
    scaler = MinMaxScaler()
    for column in x_train.columns:
        x_train[column] = scaler.fit_transform(pd.DataFrame(x_train[column]))
        x_test[column] = scaler.transform(pd.DataFrame(x_test[column]))
        ud[column] = scaler.transform(pd.DataFrame(ud[column]))

# Display normalized data
    st.write("Normalized Training Data (X_train):")
    st.dataframe(x_train.head())

    st.write("Normalized Testing Data (X_test):")
    st.dataframe(x_test.head())

    st.write("Normalized Unlabeled Data:")
    st.dataframe(ud.head())
# Display Correlation Matrix
    st.subheader("Correlation Matrix for Labeled Data:")
    correlation_matrix_labeled = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix_labeled, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, facecolor='#000000') 
    st.pyplot(fig)

    joblib.dump(scaler, 'minmax_scaler.pkl')

    st.session_state.data = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'ud': ud
    }

if __name__ == "__main__":
    main()
