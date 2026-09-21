import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Fetal Health Prediction System",
    layout="wide"
)


df = pd.read_csv("fetal_health.csv")
model = pickle.load(open("trained_model.pkl", "rb"))

st.title("Fetal Health Prediction System")

st.markdown("""
This application uses a **Random Forest Machine Learning Model**
to predict fetal health based on cardiotocography (CTG) measurements.

### Target Classes
- **1 = Normal**
- **2 = Suspect**
- **3 = Pathological**
""")


st.header("Enter Fetal Measurements")

col1, col2 = st.columns(2)

with col1:
    baseline_value = st.number_input("Baseline Value", value=120.0)
    accelerations = st.number_input("Accelerations", value=0.003)
    fetal_movement = st.number_input("Fetal Movement", value=0.01)
    uterine_contractions = st.number_input("Uterine Contractions", value=0.005)
    light_decelerations = st.number_input("Light Decelerations", value=0.001)
    severe_decelerations = st.number_input("Severe Decelerations", value=0.0)
    prolongued_decelerations = st.number_input("Prolongued Decelerations", value=0.0)
    abnormal_short_term_variability = st.number_input(
        "Abnormal Short Term Variability",
        value=30.0
    )
    mean_value_of_short_term_variability = st.number_input(
        "Mean Value of Short Term Variability",
        value=1.5
    )
    percentage_of_time_with_abnormal_long_term_variability = st.number_input(
        "Percentage of Time With Abnormal Long Term Variability",
        value=0.0
    )

with col2:
    mean_value_of_long_term_variability = st.number_input(
        "Mean Value of Long Term Variability",
        value=8.0
    )
    histogram_width = st.number_input("Histogram Width", value=70.0)
    histogram_min = st.number_input("Histogram Min", value=60.0)
    histogram_max = st.number_input("Histogram Max", value=150.0)
    histogram_number_of_peaks = st.number_input(
        "Histogram Number Of Peaks",
        value=3.0
    )
    histogram_number_of_zeroes = st.number_input(
        "Histogram Number Of Zeroes",
        value=0.0
    )
    histogram_mode = st.number_input("Histogram Mode", value=120.0)
    histogram_mean = st.number_input("Histogram Mean", value=120.0)
    histogram_median = st.number_input("Histogram Median", value=120.0)
    histogram_variance = st.number_input("Histogram Variance", value=20.0)
    histogram_tendency = st.number_input("Histogram Tendency", value=0.0)


# PREDICTION

if st.button("Predict Fetal Health"):

    input_data = [[
        baseline_value,
        accelerations,
        fetal_movement,
        uterine_contractions,
        light_decelerations,
        severe_decelerations,
        prolongued_decelerations,
        abnormal_short_term_variability,
        mean_value_of_short_term_variability,
        percentage_of_time_with_abnormal_long_term_variability,
        mean_value_of_long_term_variability,
        histogram_width,
        histogram_min,
        histogram_max,
        histogram_number_of_peaks,
        histogram_number_of_zeroes,
        histogram_mode,
        histogram_mean,
        histogram_median,
        histogram_variance,
        histogram_tendency
    ]]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.header("Prediction Result")

    if prediction == 1:
        st.success("✅ Normal Fetal Health")
        st.write(
            "The model predicts that the fetus appears healthy based on the provided measurements."
        )

    elif prediction == 2:
        st.warning("⚠️ Suspect Fetal Health")
        st.write(
            "The model detected measurements that may require additional monitoring."
        )

    else:
        st.error("🚨 Pathological Fetal Health")
        st.write(
            "The model detected measurements associated with higher-risk fetal conditions."
        )

    st.subheader("Prediction Confidence")

    confidence_df = pd.DataFrame({
        "Class": ["Normal", "Suspect", "Pathological"],
        "Probability": probabilities
    })

    st.bar_chart(confidence_df.set_index("Class"))


# PROJECT METRICS


st.header("Dataset Metrics")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Rows", len(df))

with metric2:
    st.metric("Features", len(df.columns) - 1)

with metric3:
    st.metric("Classes", 3)

with metric4:
    st.metric("Model", "Random Forest")


st.header("Dataset Visualizations")

# Class Distribution
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.countplot(x="fetal_health", data=df, ax=ax1)
ax1.set_title("Distribution of Fetal Health Classes")
st.pyplot(fig1)

st.markdown("""
**Insight:** Most fetal cases are classified as Normal,
while Suspect and Pathological cases occur less frequently.
""")

# Correlation Heatmap
st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)
ax2.set_title("Feature Correlation Heatmap")
st.pyplot(fig2)

st.markdown("""
**Insight:** The heatmap shows relationships between fetal monitoring variables.
Features with stronger correlations may contribute more strongly to prediction.
""")

# Histogram
st.subheader("Histogram of Baseline Value")

fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.histplot(df["baseline value"], bins=20, kde=True, ax=ax3)
ax3.set_title("Baseline Value Distribution")
st.pyplot(fig3)

# Boxplot
st.subheader("Baseline Value Boxplot")

fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.boxplot(x=df["baseline value"], ax=ax4)
ax4.set_title("Baseline Value Outlier Detection")
st.pyplot(fig4)