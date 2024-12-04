# music-genre-classification
A hybrid approach combining graph-based and audio-based techniques for accurate music genre classification.

**Music Genre Classification**

This repository explores the use of Machine Learning (ML) and Deep Learning (DL) techniques for music genre classification. The project utilizes the GTZAN dataset, a collection of audio clips from ten music genres.

**Dataset**

The GTZAN dataset is used for this project. It is available on Kaggle: [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

**Data Exploration and Visualization**

The code explores the dataset and visualizes various audio features:

* **Waveform:**
[Image of waveform]

* **Spectrogram:**
[Image of spectrogram]

* **Spectral Rolloff:**
[Image of spectral rolloff]

* **Chroma Features:**
[Image of chroma features]

* **Zero Crossing Rate:**
[Image of zero crossing rate]

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
