import streamlit as st
from keras.models import load_model
from Model_Prediction import pred
from matplotlib import pyplot as plt


# Load the Model
model = load_model('Brain-Tumor-Classification.h5')

# Lables
labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor found', 'Pituitary Tumor']

st.set_page_config(page_title="mr. brain",page_icon="🧠")
st.title("Brain Tumor Image Classifier")
st.write('''Welcome to Mr. Brain - Your Brain Tumor Image Classifier!

This Classifier aims to analyze MRI scans of the brain to determine the probabilities of various types of brain tumors present. 
By leveraging Convolutional Neural Network, we predict the highest probability tumor type detected in the MRI scan.
\n The chosen MRI scan gets classified into one of the following categories:

1. Glioma Tumor
2. Meningioma Tumor
3. No Tumor found
4. Pituitary Tumor

By harnessing the power of predictive analysis, With precise insights, it aids in the accurate diagnosis and tailored treatment of patients with brain tumors, enhancing the quality of Healthcare and outcomes
\n\n Upload an image and let Mr. Brain classify your brain tumor!''')
st.markdown("***")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg"],accept_multiple_files=False)

if(uploaded_file):
    st.image(uploaded_file,width=300)
    output,posibi = pred(uploaded_file)

    # Predicted Class Label
    st.write(labels[output])

    # Pie chart
    print(posibi)
    dict = {'Glioma Tumor':posibi[0][0], 'Meningioma Tumor':posibi[0][1], 'No Tumor found':posibi[0][2], 'Pituitary Tumor':posibi[0][3]}
    st.write(dict)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(posibi[0], labels=labels)
    st.pyplot(fig)