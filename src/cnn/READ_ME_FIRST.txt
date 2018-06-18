
GENERAL INFORMATION

1) ***CLONE*** the entire directory to your local workspace.

2) ***RUN*** the "dogCatCnn.py" to build, train, test the model
Changing the following parameters will lead to different runtime:

steps_per_epoch = 400
epochs = 20
validation_steps = 162
file_to_open_DOG = data_folder / "d001.jpg"

The larger the parameters, the longer runtime!

NOTES:
The testing images are in cnn/datasets/my_images/ directory 

Source: https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8