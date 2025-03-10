# BRAIN X EEG



 EEG Machine Learning Project is a Streamlit-based web application designed to analyze EEG signals, classify dominant brainwave frequencies, and provide personalized recommendations based on brain activity. The app supports both multi-channel EEG analysis and single-channel real-time EEG processing. It allows users to upload EEG data in CSV format, processes the data to compute power spectral density (PSD) using Welch’s method, and detects dominant brainwave bands such as Delta, Theta, Alpha, Beta, and Gamma. The RandomForestClassifier is used to train a model for classifying brainwave states, while StandardScaler normalizes the data for improved accuracy. The app also provides interactive visualizations, including a 3D brain model that highlights affected brain regions and a per-channel power distribution chart. Additionally, the system offers activity-based recommendations to enhance cognitive performance based on the detected brainwave states.

### **Key Features of EEG Machine Learning Project**  

#### **1. Multi-Channel EEG Analysis**  
This feature allows the system to process EEG data from multiple channels (electrodes) recorded from different brain regions. The uploaded CSV file is checked for necessary columns, and missing values are handled using mean imputation. For each EEG channel, the system computes **power spectral density (PSD)** using **Welch’s method** to estimate the power of different brainwave frequency bands. The dominant brainwave is then identified based on the highest normalized power value.

#### **2. Single-Channel Real-Time Analysis**  
For users who have only a single EEG signal channel, the app provides real-time analysis of EEG data. After uploading a single-channel EEG file, the system computes the power in different frequency bands and determines the dominant brainwave state. The user receives immediate feedback on their cognitive state and personalized recommendations. Additionally, a **line chart visualization** helps users observe signal fluctuations over time.

#### **3. Machine Learning-Based Classification**  
The project integrates a **RandomForestClassifier** to classify brainwave states based on the computed bandpower. The classifier is trained using labeled EEG data, and it helps in predicting the most dominant mental state. The data is split into training and testing sets using **train_test_split**, and **StandardScaler** is applied to normalize the feature values for better accuracy. The model's performance is evaluated using a **classification report**, which includes accuracy, precision, recall, and F1-score metrics.

#### **4. 3D Brain Visualization of Affected Regions**  
To provide an intuitive understanding of brain activity, the project includes a **3D brain model visualization** using **Plotly**. The EEG channels corresponding to different brain regions are mapped to **3D coordinates**, and the dominant frequency bands detected in each channel are visually represented. Channels with significant activity in a particular frequency band are highlighted, helping users identify which parts of the brain are most active.

#### **5. Channel-Wise Power Distribution Analysis**  
For each selected subject, the app displays the power distribution of different EEG channels. This feature enables users to compare how different brain regions contribute to overall cognitive activity. The **power distribution graphs** illustrate the variations in bandpower across different channels, providing insights into which parts of the brain are most dominant for specific tasks.

#### **6. Personalized Recommendations Based on Brainwave Activity**  
Once the dominant brainwave frequency is identified, the system provides personalized recommendations to optimize mental performance. For example:  
- **Alpha waves (8-13 Hz):** Suggests relaxation techniques like meditation.  
- **Beta waves (13-30 Hz):** Recommends engaging in analytical or problem-solving tasks.  
- **Gamma waves (30-50 Hz):** Encourages participation in complex cognitive tasks.  
- **Theta waves (4-8 Hz):** Suggests engaging in creative activities or daydreaming.  
- **Delta waves (0.5-4 Hz):** Advises prioritizing sleep and relaxation.  

#### **7. Interactive Web-Based Interface**  
The entire analysis process is packaged into an easy-to-use **Streamlit web application**, allowing users to upload their EEG data, visualize the results, and receive insights without requiring advanced technical knowledge. The UI includes multiple tabs for viewing recommendations, 3D visualizations, and channel-wise power distribution, ensuring a seamless user experience.

#### **8. Data Preprocessing and Normalization**  
To ensure data quality, the system preprocesses the EEG data before analysis. It performs:  
- **Missing value handling:** Fills gaps in data using mean imputation.  
- **Feature scaling:** Uses **StandardScaler** to normalize the power values, ensuring fair comparisons between different subjects.  
- **Data validation:** Ensures that the uploaded CSV file contains the required columns, preventing errors during analysis.  

By incorporating these key features, your project provides a **comprehensive EEG analysis tool** that can be used for **brainwave classification, cognitive insights, and real-time recommendations** for mental well-being. 
### **How  EEG Analysis Project Works**  

1. **Upload EEG Data** – User uploads a CSV file with brainwave recordings.  
2. **Data Cleaning** – Missing values are filled, and invalid data is removed.  
3. **Power Calculation** – Brainwave power is computed for different frequency bands (Delta, Theta, Alpha, Beta, Gamma).  
4. **Find Dominant Brainwave** – The most active brainwave is identified.  
5. **Machine Learning Prediction** – A trained model classifies the brainwave state.  
6. **3D Brain Visualization** – Displays affected brain regions using a 3D model.  
7. **Power Distribution Graphs** – Shows EEG signal strength in different brain regions.  
8. **Personalized Recommendations** – Suggests activities based on dominant brainwaves.  
9. **Interactive UI** – Users can explore results and insights through a web app.
### **Technologies Used in EEG Analysis Project**  

1. **Programming Language:**  
   - Python 🐍 (for data processing, machine learning, and visualization)  

2. **Data Processing & Analysis:**  
   - **pandas** – Handles EEG data from CSV files  
   - **numpy** – Performs numerical computations  
   - **scipy.signal (Welch’s Method)** – Computes power spectral density (PSD)  

3. **Machine Learning:**  
   - **scikit-learn (RandomForestClassifier)** – Classifies brainwave states  
   - **StandardScaler** – Normalizes EEG feature values  

4. **Data Visualization:**  
   - **plotly (Express & Graph Objects)** – Creates interactive graphs and 3D brain models  
   - **Streamlit** – Provides a web-based interface for users  

5. **Web Framework:**  
   - **Streamlit** – Builds an interactive web application for EEG analysis  

6. **File Handling:**  
   - **CSV file processing** – Reads multi-channel EEG data for analysis

  ![Image](https://github.com/user-attachments/assets/58bcdc6f-5b6d-47e7-8e68-a0ba1ef84b3b)
