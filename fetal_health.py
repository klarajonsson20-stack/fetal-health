"""AI Disclosure Template
Tool: ChatGPT-5
Pupose: Implement weighted feature importance calculation and plotting for Soft Voting Classifier; 
color code predicted classes in Streamlit app; display prediction probabilities with one decimal place"""

# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")   

st.title("Fetal Health Prediction")
st.write("Predict fetal health status using machine learning based on cardiotocography (CTG) data.")

gif_path = "fetal_health_image.gif"
st.image(gif_path, use_container_width=True)


# Add app description section
st.write("Utilize our advanced machine learning application to predict fetal healt classification")

# Load models
@st.cache_resource
def load_model(path="fetal_health_models.pickle"):  
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()   
    
# Load default raw data
@st.cache_resource
def load_default_raw(path="fetal_health_train_raw.csv"):
    return pd.read_csv(path)

default_X_raw = load_default_raw()
feature_cols = list(default_X_raw.columns)  # pre made training columns

# Create a sidebar for csv upload
st.sidebar.subheader('**Fetal Health Features input**')
uploaded_file = st.sidebar.file_uploader("Upload your data", type=["csv"])
st.sidebar.warning("**⚠️ Ensure your CSV file strictly follows the format outlined below**")

# Create sample dataframe for user reference and fill 5 rows with data from default_X_raw
sample_df = default_X_raw[feature_cols].head(5)
st.sidebar.dataframe(sample_df, use_container_width=True)

# Let the user pick between models
model_choice = st.sidebar.radio("Choose model for prediction", ("Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"))

# Predict fetal health status
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.success("**✅ CSV file uploaded successfully!**")

    # Predict without button click
    if model_choice == "Random Forest":
        selected_model = model['Random Forest']
    elif model_choice == "Decision Tree":
        selected_model = model['Decision Tree']
    elif model_choice == "AdaBoost":
        selected_model = model['AdaBoost']
    else:
        selected_model = model['Soft Voting']
    
    st.sidebar.info(f"**✔️ You selected: {model_choice}**")


    X = input_df[feature_cols]
    proba = selected_model.predict_proba(X)
    predictions = selected_model.predict(X)
    
    input_df['Predicted Fetal Health'] = pd.Series(predictions)
    # Display prediction probabilities in a new column with one decimal place
    input_df['Prediction Probability (%)'] = pd.Series(proba.max(axis=1) * 100).map("{:.1f}".format)
    st.subheader("Predicting Fetal Health Class Using " + model_choice + " Model")

    # Color cells based on class name
    # ChatGPT-5 suggested code for color coding   
    colormap_df = input_df.style.applymap(
        lambda v:
            "background-color: lime" if v == "Normal"
            else "background-color: yellow" if v == "Suspect"
            else "background-color: orange" if v == "Pathological"
            else "",
        subset=['Predicted Fetal Health'])
    
    st.dataframe(colormap_df, use_container_width=True)
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Feature Importance Plot", "Confusion Matrix", "Classification Report"])
    st.write("Classification report: Precision, Recall, F1-Score and Support for each health condition.")
    with tab1:
        st.write("### Feature Importance Plot")
        if model_choice == "Random Forest":
            st.image('rf_feature_importance.svg')
        elif model_choice == "Decision Tree":
            st.image('dt_feature_importance.svg')
        elif model_choice == "AdaBoost":
            st.image('ada_feature_importance.svg')
        else:
            st.image('vote_feature_importance.svg')
    with tab2:
        st.write("### Confusion Matrix")
        if model_choice == "Random Forest":
            st.image('rf_confusion_matrix.svg')
        elif model_choice == "Decision Tree":
            st.image('dt_confusion_matrix.svg')
        elif model_choice == "AdaBoost":
            st.image('ada_confusion_matrix.svg')
        else:
            st.image('vote_confusion_matrix.svg')
    with tab3:
        st.write("### Classification Report")
        if model_choice == "Random Forest":
            report_df = pd.read_csv('rf_classification_report.csv', index_col = 0).transpose()
            cmap = "PuBu"
        elif model_choice == "Decision Tree":
            report_df = pd.read_csv('dt_classification_report.csv', index_col = 0).transpose()
            cmap = "PuRd"
        elif model_choice == "AdaBoost":
            report_df = pd.read_csv('ada_classification_report.csv', index_col = 0).transpose()
            cmap = "Oranges"
        else:
            report_df = pd.read_csv('vote_classification_report.csv', index_col = 0).transpose()
            cmap = "YlGn"
        
        # Apply unique background gradient color per model
        styled_report = report_df.style.background_gradient(cmap=cmap).format(precision=2)
        st.dataframe(styled_report, use_container_width=True)

else:
    st.info("***ℹ️ Please upload data to proceed***")





