# EEG-ml-project
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Streamlit page configuration
st.set_page_config(page_title="EEG Analysis App", layout="wide")

# Placeholder for your bandpower calculation function
def compute_bandpower(data, freq_range, fs):
    # Dummy implementation - replace with your actual bandpower calculation
    return np.random.random()

# Function to generate recommendations based on dominant bands
def generate_recommendations(dominant_band):
    recommendations = {
        'Alpha': "Engage in activities that require calm focus, such as meditation.",
        'Beta': "Participate in problem-solving tasks to boost alertness.",
        'Gamma': "Consider engaging in complex cognitive tasks for enhanced brain function.",
        'Theta': "Incorporate creative tasks or daydreaming for relaxation.",
        'Delta': "Prioritize sleep and recovery for overall well-being."
    }
    return recommendations.get(dominant_band, "No recommendations available.")

# Function to load and preprocess the dataset (multi-channel)
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Subject' not in df.columns:
        st.error("The uploaded CSV must contain a 'Subject' column.")
        st.stop()
    
    subjects = df['Subject'].unique()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df, numeric_cols, subjects

# Function to compute band power using Welch's method
def compute_bandpower_welch(data, band, fs):
    freqs, psd = welch(data, fs, nperseg=1024)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[idx_band])

# Function to get brain regions dynamically based on available channels
def get_brain_regions():
    brain_regions = {
        "Fp1": {"region": "Frontal", "coords": (-1, 1, 1)},
        "Fp2": {"region": "Frontal", "coords": (1, 1, 1)},
        "F3": {"region": "Frontal", "coords": (-1, 0, 1)},
        "F4": {"region": "Frontal", "coords": (1, 0, 1)},
        "C3": {"region": "Central", "coords": (-1, -1, 0)},
        "C4": {"region": "Central", "coords": (1, -1, 0)},
        "P3": {"region": "Parietal", "coords": (-1, -2, -1)},
        "P4": {"region": "Parietal", "coords": (1, -2, -1)},
        "O1": {"region": "Occipital", "coords": (-1, -3, -2)},
        "O2": {"region": "Occipital", "coords": (1, -3, -2)},
        "T3": {"region": "Temporal", "coords": (-2, 0, 0)},
        "T4": {"region": "Temporal", "coords": (2, 0, 0)},
    }
    return brain_regions

# Function to train and evaluate the machine learning model (multi-channel)
def train_model(df, numeric_cols, subjects, brain_regions):
    frequency_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }

    fs = 256
    subject_results = []
    subject_labels = []
    subject_dominant_bands = {}

    for subject in subjects:
        subject_data = df[df['Subject'] == subject]
        subject_channel_results = []
        subject_labels_results = []
        subject_channel_bands = {}

        for channel in numeric_cols:
            if channel not in brain_regions:
                continue

            channel_data = subject_data[channel].values
            band_powers = {}

            for band_name, band_freq in frequency_bands.items():
                band_powers[band_name] = compute_bandpower_welch(channel_data, band_freq, fs)

            total_power = sum(band_powers.values())
            normalized_powers = {band: power / total_power for band, power in band_powers.items()}

            threshold = 0.15
            dominant_bands = [band for band, power in normalized_powers.items() if power >= threshold]
            dominant_band = (dominant_bands[0] if len(dominant_bands) == 1 
                             else max(normalized_powers, key=normalized_powers.get))

            subject_channel_results.append(list(band_powers.values()))
            subject_labels_results.append(dominant_band)

            subject_channel_bands[channel] = dominant_band

        if not subject_channel_results:
            continue

        subject_results.append(pd.DataFrame(subject_channel_results, columns=frequency_bands.keys()))
        subject_labels.append(pd.Series(subject_labels_results))

        subject_dominant_bands[subject] = subject_channel_bands

    if not subject_results:
        st.error("No valid data found for the mapped brain regions.")
        st.stop()

    X = pd.concat(subject_results).reset_index(drop=True)
    y = pd.concat(subject_labels).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return report_df, subject_dominant_bands

# Function for single-channel EEG analysis
def analyze_single_channel(uploaded_file):
    df = pd.read_csv(uploaded_file)

    if 'EEG Signal' not in df.columns:
        st.error("The uploaded file must contain an 'EEG Signal' column.")
        return

    eeg_signal = df['EEG Signal'].values
    st.write("Computing bandpower... (Placeholder)")
    bandpower = compute_bandpower(eeg_signal, (8, 13), 250)
    st.write(f"Computed Bandpower: {bandpower}")

    dominant_band = 'Alpha'  # This should be determined based on your logic
    recommendation = generate_recommendations(dominant_band)
    st.write(f"*Recommendations based on {dominant_band} state:* {recommendation}")

    if st.button("Generate EEG Visualization"):
        st.line_chart(df.set_index('Timestamp'))

# Function to plot the 3D brain model visualization with per-channel band highlighting
def plot_3d_brain(brain_regions, subject_dominant_bands, selected_subject, mental_states):
    fig = go.Figure()
    for channel, info in brain_regions.items():
        x, y, z = info["coords"]
        dominant_band = subject_dominant_bands.get(selected_subject, {}).get(channel, None)
        
        # Set color to red for affected parts, otherwise gray
        color = 'red' if dominant_band else 'gray'
        
        if dominant_band:
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                text=f"{channel} ({info['region']}) - {dominant_band}: {mental_states.get(dominant_band, 'Unknown state')}",
                mode='markers+text',
                marker=dict(size=8, color=color, opacity=0.8),
                name=channel
            ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=0.7),
            camera_eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        title=f"3D Brain Visualization for Subject {selected_subject}",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    st.plotly_chart(fig)

# Function to plot the power distribution of each affected channel for a selected subject
def plot_affected_power_distribution(subject_dominant_bands, df, selected_subject, numeric_cols):
    if selected_subject not in subject_dominant_bands:
        st.error(f"No dominant band information available for Subject {selected_subject}.")
        return

    subject_data = df[df['Subject'] == selected_subject]
    channel_band_powers = subject_dominant_bands[selected_subject]
    st.write(f"### Affected Channel-wise Power Distribution for Subject {selected_subject}")

    for channel in numeric_cols:
        if channel in channel_band_powers:
            st.write(f"*Channel:* {channel} - *Dominant Band:* {channel_band_powers[channel]}")
            st.bar_chart(subject_data[channel])

# Main function to control the flow of the app
def main():
    st.title("EEG Analysis App")

    analysis_option = st.selectbox("Select Analysis Type", ["Multi-Channel Analysis", "Real-Time Single Channel Analysis"])

    if analysis_option == "Multi-Channel Analysis":
        uploaded_file = st.file_uploader("Upload Multi-Channel CSV", type=["csv"])
        if uploaded_file:
            df, numeric_cols, subjects = load_and_preprocess_data(uploaded_file)
            st.write("### Model Training and Evaluation")
            report_df, subject_dominant_bands = train_model(df, numeric_cols, subjects, get_brain_regions())
            st.dataframe(report_df)

            selected_subject = st.selectbox("Select Subject for Visualization", subjects)
            brain_regions = get_brain_regions()

            tab1, tab2, tab3 = st.tabs(["Recommendations", "3D Brain Visualization", "Channel-wise Power Distribution"])
            
            with tab1:
                dominant_bands = subject_dominant_bands[selected_subject]
                recommendations = {band: generate_recommendations(band) for band in dominant_bands.values()}
                st.write("### Recommendations")
                for band, rec in recommendations.items():
                    st.write(f"- {band}: {rec}")

            with tab2:
                st.write("### 3D Brain Visualization")
                plot_3d_brain(brain_regions, subject_dominant_bands, selected_subject, {})

            with tab3:
                st.write("### Channel-wise Power Distribution")
                plot_affected_power_distribution(subject_dominant_bands, df, selected_subject, numeric_cols)

    elif analysis_option == "Real-Time Single Channel Analysis":
        uploaded_file = st.file_uploader("Upload Single Channel CSV", type=["csv"])
        if uploaded_file:
            analyze_single_channel(uploaded_file)

if _name_ == "_main_":
    main()
