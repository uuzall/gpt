# This repo implements the Generative Pre-Trained Transformer (GPT) from scratch 

It is heavily guided by Andrej Karpathy's video (https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3354s&ab_channel=AndrejKarpathy). Although, instead of using the tiny-Shakespeare dataset like he did, I created my own dataset using the transcripts of the show Friends. 

* The data.py file handles all the importing, encoding, and decoding of the dataset. 
* The gpt.py file includes the GPT model. 
* The main.ipynb notebook contains the training loop. 
* The testing_gpt.ipynb notebook contains the testing with generations I did with the trained model. 

Apart from the files, there are several folders: 
* The 'data' folder contains the data. 
* The 'generations' folder contains all the generations saved from 'x' intervals of training. You can go through them to see how good they are at each interval. 
* If you use this repo, you should create a 'models' folder that stores the best performing model (based on the lowest loss in the validation dataset). 

P.S. I do not remember why I saved the 'val_losses.pkl' file but just know that it's there. 