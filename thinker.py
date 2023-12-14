from fastai.vision.all import *

learner = load_learner("export.pkl")
is_dog,_,probs = learner.predict(PILImage.create("dataset/single_prediction/cat_or_dog_1.jpg"))

print(f"This is a: {is_dog}.")
print(f"Probability it's a dog: {probs[0]:.4f}")