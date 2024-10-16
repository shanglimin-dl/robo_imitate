# Imitation Learning

<div align='center'>
    <img src="./media/training_and_evaluation.png">
</div>

</br>

Inside this folder, you can find scripts for simple Diffusion model training and evaluation. During model training, we provide the model with images, the current pose, and actions performed by an expert during evaluation. The input data is normalized before being used for learning behavior. In the end, the model learns to generate actions similar to those of the expert by using the images and the current state obtained from observations.

## Model training
Inside [config.py](imitation/common/config.py) script you can change default config params. When you prepair data you can start with training model.