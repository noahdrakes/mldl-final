# MultiModal Violdence Detection Model 
from XD-Violence paper

## Dataset

Here is the link to the zipped Dataset and pretrained model: https://drive.google.com/file/d/1BbYIWZ1O_K82KkHMJL2WEc3TcxOXiwM1/view?usp=drive_link.
I unzip this folder in google collab to access the file for the model

*****************************
Gus Edits: see my branch

Added a training pipeline at the bottom of the notebook. This setup extracts features from the VAE to pair with labels for input to the HL-net
Added a chunk with a bunch of changes to args, dataset class to potentially consider creating a validation split for the data, we can choose to incorporate these or ignore them

*****************************
Gus Edits: See my branch

Added validation splitting to the data. Can not create test/train/val splits for rbg, aud, and mix2 modalities

flow not yet supported, but see my notes in the code about how to implement it. it would be kinda simple

Run the labeled chunks to generate data splits. This involved making changes to the Args, Dataset classes as well as rewriting some of those functions that were called like "Make List" or "Update File List Paths". Those chunks are now gone and replaced. Proceed with caution. if code you wrote in the past depended on these, there's a chance they may no longer work, but you can check what i wrote in my val splitter functions to try and resolve things. everything should still be there in a somewhat similar form.



******************************
LINKS TO FULLY TRAINED VAE

look for best_trained_vae.pkl or the latest .pt file

https://drive.google.com/file/d/17gNDWn7ma2n0ZQ915PtpyDsdRR9fbei4/view?usp=sharing
