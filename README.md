![GitHub Repo stars](https://img.shields.io/github/stars/BirukBelihu/FaceMaskDetector)
![GitHub forks](https://img.shields.io/github/forks/BirukBelihu/FaceMaskDetector)
![GitHub issues](https://img.shields.io/github/issues/BirukBelihu/FaceMaskDetector)
![GitHub license](https://img.shields.io/github/license/BirukBelihu/FaceMaskDetector)

# Moodly

A Simple Facial Emotion Recognition System In A Live Camera Using Computer Vision & Machine Learning.

# Running

To Get Started With Moodly On Your Local Machine Follow This Simple Steps One By One To Get Up & Running.

Make Sure You Have **[Git](https://git-scm.com/)** & **[Python](https://python.org)** Installed On Your Machine.

```
git --version
```

```
python --version
```

# Reminder
Make Sure You're Using **Python Version 3.9-3.12**.

Clone The Repository

```
git clone https://github.com/BirukBelihu/Moodly.git
```

Go Inside The Project

```
cd Moodly
```

Install Required Dependencies

```
pip install -r requirements.txt
```

Run Moodly
```
python main.py
```

# Dataset
The Dataset Used To Train The Model Is [face-expression-recognition-dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) Dataset. It Contains More Than 35K Face Images With 7 Different Emotions(angry, disgust, fear, happy, neutral, sad, surprise) In Pixel Grayscale.

# Training
To Train Your Own Model You Can Use The Kaggle Notebook From The notebook Folder Or Run ```model_trainer.py```.

```
python model_trainer.py
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
