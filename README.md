# A Perceptual Prediction Framework for Self Supervised Event Segmentation
Code for the paper A Perceptual Prediction Framework for Self Supervised Event Segmentation

## Preparation:
  All videos must be extracted into individual frames. One folder per video with each folder containing the frames for the video.
    Example code for extracting frames is shown in preprocessVideo.py for the Breakfast Actions dataset.
  Requirements: Python 2.7, Tensorflow 1.6+, scipy and numpy
  A JSON file with th video names to process as keys needs to be created.
  
  
## Training
  Run Zacks_VGG_RNN.py for RNN based prediction model
  Run Zacks_VGG_RNN_EventLayer.py for LSTM based prediction model
  
  Arguments for the program are the same:
    python Zacks_VGG_RNN.py <jsonData path> <video frames root directory> <path to save model> <use active learning (1) or not (0)>
  
## Inference
  Run Zacks_VGG_RNN_Inference.py for RNN based prediction model
  Run Zacks_VGG_RNN_EventLayer_Inference.py for LSTM based prediction model
  
  Arguments for the program are the same:
    python Zacks_VGG_RNN.py <jsonData path> <video frames root directory> <path to restore model> <output file name to write loss characteristics>

  An output text file will be written into each folder with the prediction loss at each time instance t.
  
  Run the getBoundaries.py script to transform the loss file into boundaries and visualize the predictions.
  
## Evaluation Remarks
  Note: Due to the self-supervised nature of the approach, the output classes will be in serial order (i.e. 0,1, ..., n). TO evaluate, transform ground truth to same format and run evaluation scripts for the dataset. 
  Example Evaluation for Breakfast Actions Dataset is shown in evaluateBreakfastActions.py.
  Note: The original script was written in Tensorflow 1.3. Newer versions of Tensorflow and Python have shown minor variations in performance. Running the script should lead to the reported accuracy of 42.8% MoF on Breakfast Actions.
