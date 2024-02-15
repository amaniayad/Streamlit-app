import streamlit as st

def main():
    st.title("Mini Project SEDS - Aerodelay")
    st.markdown("---")

    # General Information
    st.subheader("Higher School of Computer Science - Sidi Bel Abbes")

    # Group Presentation
    st.subheader("Group")
    st.write("**Members:** AYAD Amani, BELMILOUD Maroua, SENKADI Khawla")
    st.write("**Specialization:** IASD")
    st.write("**Groups:** Group 1, Group 2")

    # Project Information
    st.subheader("Project")
    st.write("**Title:** Flight Delay Prediction")
    st.write("**Description:** The project aims to develop a predictive model for flight delays using machine learning techniques.")

    # Academic Year
    st.write("**Academic Year:** 2024")

    # Streamlit Information
    st.markdown("---")
    st.header("About Streamlit")
    st.write("Streamlit is an open-source Python library designed to simplify the process of creating interactive web applications directly from Python scripts.")
    st.write("It provides a user-friendly interface that allows developers to quickly build and customize applications using familiar Python syntax.")
    st.write("For more information, you can visit the official [Streamlit website](https://streamlit.io/).")

    # Cloud Deployment Information
    st.markdown("---")
    st.header("Cloud Deployment")
    st.write("Once the Streamlit application is developed, it can be deployed to cloud platforms for public access.")
    st.write("Popular cloud platforms for deployment include Heroku, AWS, and Google Cloud Platform.")

if __name__ == "__main__":
    main()
