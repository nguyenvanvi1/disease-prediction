# Plant Disease Diagnosis Application

## Overview
This is a mobile application designed to help farmers and plant enthusiasts diagnose plant diseases using image recognition powered by Artificial Intelligence (AI). 
The app allows users to upload photos of plant leaves, receives predictions about potential diseases, and provides detailed information such as symptoms, solutions, and impacts.
The application is built with a React Native frontend and a FastAPI backend, integrated with a pre-trained deep learning model using TensorFlow/Keras.

### Key Features
- **Image-Based Diagnosis**: Upload a photo of a leaf to get a disease prediction with confidence score.
- **Detailed Information**: View descriptions, symptoms, solutions, and impacts for identified diseases.
- **History Management**: Save and review diagnosis history, with the ability to delete entries.
- **User Authentication**: Secure access with JWT-based authentication.
- **Offline Support**: History is stored locally using SQLite.

## Technologies Used
- **Frontend**: React Native (with components like ScrollView, ImagePicker, and AsyncStorage).
- **Backend**: FastAPI (with asynchronous API endpoints).
- **AI Model**: MobileNet (a lightweight Convolutional Neural Network) implemented with TensorFlow/Keras.
- **Database**: SQLite for local storage of history.
- **Other Tools**: Axios for API calls, Pytz for timezone handling, and Python libraries for image processing.

## Feature
Register/Login: Use the app to register a new account or log in with existing credentials.
![image](https://github.com/user-attachments/assets/a27cadad-2820-49ef-8295-3a42a281d083)

Upload Image: Select or capture a photo of a plant leaf via the "Chọn Ảnh" button.
![image](https://github.com/user-attachments/assets/ea91698f-0a15-44cf-96fd-4154c86dd0ed)

View Results: The app will display the predicted disease name, confidence score, description, symptoms, solutions, and impact.
![image](https://github.com/user-attachments/assets/5af56ff9-bb5e-4057-9037-323c66470922)

Manage History: Access the history screen to review past diagnoses and delete entries if needed.
![image](https://github.com/user-attachments/assets/0c5bb71c-2201-49ce-9734-f7783308368b)
![image](https://github.com/user-attachments/assets/792ddc91-0662-4ad8-acf3-f1d50a5963b3)
