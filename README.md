# MultiModal Violdence Detection Model 
from XD-Violence paper

## Dataset

Here is the link to the zipped Dataset and pretrained model: https://drive.google.com/file/d/1BbYIWZ1O_K82KkHMJL2WEc3TcxOXiwM1/view?usp=drive_link.
I unzip this folder in google collab to access the file for the model

*****************************
Gus Edits: see my branch

Added a training pipeline at the bottom of the notebook. This setup extracts features from the VAE to pair with labels for input to the HL-net
Added a chunk with a bunch of changes to args, dataset class to potentially consider creating a validation split for the data, we can choose to incorporate these or ignore them
