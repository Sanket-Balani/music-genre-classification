**Music Genre Classification**
A hybrid approach combining graph-based and audio-based techniques for accurate music genre classification.

This repository explores the use of Machine Learning (ML) and Deep Learning (DL) techniques for music genre classification. The project utilizes the GTZAN dataset, a collection of audio clips from ten music genres.

**Dataset**

The GTZAN dataset is used for this project. It is available on Kaggle: [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

**Data Exploration and Visualization**

The code explores the dataset and visualizes various audio features:

* **Waveform:**
![image](https://github.com/user-attachments/assets/f5b3ee88-4fc4-403c-8a66-dd755572d536)


* **Spectrogram:**
![image](https://github.com/user-attachments/assets/5681d41b-28c0-4db6-b6f3-dc90408254ef)
![image](https://github.com/user-attachments/assets/e028aecd-b7b3-44fa-8502-350b15bc4009)
The vertical axis represents frequencies (from 0 to 10kHz), and the horizontal axis represents the time of the clip.

* **Spectral Rolloff:**
![image](https://github.com/user-attachments/assets/fddd685b-e957-49dc-89a9-fac3ff64acd7)


* **Chroma Features:**
![image](https://github.com/user-attachments/assets/083284cc-ca2e-4108-83a0-28c911de34e7)



* **Zero Crossing Rate:**
![image](https://github.com/user-attachments/assets/a7cc1093-fb43-4a01-bc3e-17497ce0a0dc)


**Model Building and Training**

* **CNN Model:**
    * A Convolutional Neural Network (CNN) model is built to extract features from audio spectrograms.
    * The model is trained on the GTZAN dataset and evaluated on a test set.
* **SVM Model:**
    * A Support Vector Machine (SVM) model is used for classification based on extracted features.
    * The SVM model is trained and evaluated on the dataset.

**Prediction on New Songs**

* New audio files are preprocessed to extract features.
* The trained model is used to predict the genre of the new song.

**Results and Future Work**

* The CNN model has shown promising results in classifying music genres.
* Future work involves exploring different model architectures, data augmentation techniques, and fine-tuning hyperparameters.

**To run the code:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/music_genre_classification.git
   ```
2. **Set up the environment:**
   Ensure you have the required libraries installed (pandas, numpy, librosa, scikit-learn, TensorFlow, etc.).
   Download and install the GTZAN dataset.
3. **Run the code:**
   Execute the Python script to train the models, make predictions, and visualize results.

**Additional Considerations:**

* Experiment with different hyperparameters like learning rate, batch size, and number of epochs.
* Consider data augmentation techniques to improve model performance.
* Explore more advanced CNN architectures like VGG or ResNet.

By following these steps and considering the potential improvements, you can build an effective music genre classification system.
