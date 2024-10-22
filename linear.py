import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from pymongo import MongoClient
import json

# Sample data (Hours, Marks)
data = {
    'Hours': [0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11],
    'Marks': [34, 36, 33, 39, 42, 45, 38, 45, 53, 46, 56, 59, 55, 56, 72, 59, 62, 71, 78, 88, 61, 74, 71, 89, 82, 67, 89, 81, 82, 79]
}

# Create DataFrame from the sample data
df = pd.DataFrame(data)

# Prepare the model
X = df[['Hours']]  # Predictor (Study hours)
y = df['Marks']    # Target (Marks)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Function to predict marks based on study hours
def predict_marks(hours):
    return model.predict(np.array([[hours]]))[0]

# MongoDB connection setup
try:
    client = MongoClient("mongodb+srv://viswa:6374353499@cluster0.zrpec.mongodb.net/")
    db = client["student_db"]  # Database name
    collection = db["marks_predictions"]  # Collection name
    st.success("Connected to MongoDB successfully")
except Exception as e:
    st.error(f"Error connecting to MongoDB: {e}")

# Streamlit App
st.title("Student Marks Prediction Based on Study Hours")

# Get user input for student details
student_name = st.text_input("Enter student name:")
course = st.text_input("Enter course:")

# Get user input for study hours for 3 subjects using sliders
subject_1 = st.text_input("Enter subject 1 name:", "Subject 1")
hours_1 = st.slider(f"Select study hours for {subject_1}", 0.0, 12.0, 1.0)

subject_2 = st.text_input("Enter subject 2 name:", "Subject 2")
hours_2 = st.slider(f"Select study hours for {subject_2}", 0.0, 12.0, 1.0)

subject_3 = st.text_input("Enter subject 3 name:", "Subject 3")
hours_3 = st.slider(f"Select study hours for {subject_3}", 0.0, 12.0, 1.0)

# Predict marks for each subject and store in MongoDB
if student_name and course:
    marks_1 = predict_marks(hours_1)
    marks_2 = predict_marks(hours_2)
    marks_3 = predict_marks(hours_3)

    # Store prediction data in MongoDB
    record = {
        "student_name": student_name,
        "course": course,
        "predictions": {
            subject_1: {
                "hours": hours_1,
                "predicted_marks": marks_1
            },
            subject_2: {
                "hours": hours_2,
                "predicted_marks": marks_2
            },
            subject_3: {
                "hours": hours_3,
                "predicted_marks": marks_3
            }
        }
    }

    try:
        collection.insert_one(record)  # Insert record into MongoDB
        st.success("Record inserted into MongoDB successfully!")
    except Exception as e:
        st.error(f"Error inserting record into MongoDB: {e}")

    # Display results
    st.write(f"\nPrediction for **{student_name}** ({course}):")
    st.write(f"{subject_1}: **{marks_1:.2f}** marks (Based on {hours_1} study hours)")
    st.write(f"{subject_2}: **{marks_2:.2f}** marks (Based on {hours_2} study hours)")
    st.write(f"{subject_3}: **{marks_3:.2f}** marks (Based on {hours_3} study hours)")

    # Create a 3D scatter line plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df['Hours'],  # Use the original Hours data
        y=df['Marks'],  # Use the original Marks data
        z=np.zeros(len(df['Marks'])),  # Z-axis can be zeros if you want to flatten it
        mode='lines+markers',  # Show both lines and markers
        marker=dict(
            size=10,
            color=df['Marks'],  # Color by Marks
            colorscale='Viridis',  # Color scheme
            opacity=0.8
        ),
        line=dict(
            width=2,  # Width of the line connecting points
            color='blue'  # Color of the connecting lines
        )
    )])

    # Add axis labels
    fig.update_layout(scene=dict(
        xaxis_title='Hours',
        yaxis_title='Marks',
        zaxis_title='Z-axis (not used)',  # Z-axis title
        zaxis=dict(showgrid=False)  # Optionally hide the grid for the Z-axis
    ))

    # Display the 3D plot in Streamlit
    st.plotly_chart(fig)

    # Matplotlib Scatter Line Plot
    fig2, ax = plt.subplots()
    ax.plot([hours_1, hours_2, hours_3], [marks_1, marks_2, marks_3], marker='o', linestyle='-', color='b', label='Predicted Marks')
    ax.scatter([hours_1, hours_2, hours_3], [marks_1, marks_2, marks_3], color='red')  # Add scatter points in red

    # Set plot labels and title
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Predicted Marks")
    ax.set_title(f"Study Hours vs Predicted Marks for {student_name}")
    ax.legend()

    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig2)
