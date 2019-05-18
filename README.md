# Spectral Autoencoder

This is a Keras implementation of the baseline spectral autoencoder described in google deepmind <a href="https://arxiv.org/pdf/1704.01279.pdf"> paper </a>.

You can have a look at our results here:
<li>Note reconstruction: [Link yt]
<li>Audio generation with VAE: [Link yt]

## Dataset

We used a subset of the public <a href="https://magenta.tensorflow.org/datasets/nsynth">Nsynth</a> dataset composed by *brasses* and *flutes*. We got the log-magnitude spectra of each audio and we used them as *input/target* during training process.
As mentioned in the original article we used <a href="https://ieeexplore.ieee.org/document/1164317">Griffin & Lim</a> algorithm to reconstruct the phase of each signal.
 
## Implementation
In order to reduce the huge number of parameters of the original model we achieved a dimensionality reduction of the filters with respect of it.
(TODO: base model description)

We also implemented a variational version of the baseline autoencoder to see if a meaningful audio generation was possible in this case. Even in this case the phase was reconstructed using Griffin & Lim algorithm

## Requirements

| Package   |      Version     | 
|----------|:-------------:|
| Numpy |  1.15.4 | 
| Keras |    2.2.4   |   
| Tensorflow | 1.6.0 |   
| Librosa | 0.6.2 |  

## License
The code is released under the terms of TODO license.
