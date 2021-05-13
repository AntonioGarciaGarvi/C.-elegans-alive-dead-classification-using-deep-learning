# *C. elegans* alive/dead classification using deep learning
## Introduction
The automation of lifespan assays with *C. elegans* cultured in standard Petri dishes is a challenging problem, as there are several problems that hinder detection: occlusions at the edges of the plate, accumulation of dirt, aggregations of worms, etc. In addition, determining whether a worm is alive or dead can be complex as the last few days they hardly move.

In this project we propose a method that combines traditional computer vision techniques with an alive/dead *C. elegans* classifier based on convolutional and recurrent neural networks from low-resolution images. 
![GitHub Logo](https://github.com/AntonioGarciaGarvi/C.-elegans-alive-dead-classification-using-deep-learning/blob/main/NN.bmp)

## Alive or dead *C. elegans* classification demo
This repository includes a demo that allows you to see how our model is able to classify a *C. elegans* as alive or dead using a sequence of three images corresponding to the current day, the day before and the day after.

Run the demo in [google colab]

## Image adquisition system:
Images were captured by an [open hardware system](https://github.com/JCPuchalt/c-elegans_smartLight).

## References 
* Puchalt, J. C., Sánchez-Salmerón, A.-J., Martorell Guerola, P. & Genovés Martínez, S. "Active backlight for automating visual monitoring: An analysis of a lighting control technique for Caenorhabditis elegans cultured on standard Petri plates". PLOS ONE 14.4 (2019) [doi](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215548).
* Puchalt, J.C., Sánchez-Salmerón, A., Ivorra, E. et al. "Improving lifespan automation for Caenorhabditis elegans by using image processing and a post-processing adaptive data filter". Scientific Reports (2020) [doi](https://www.nature.com/articles/s41598-020-65619-4).
* Layana Castro Pablo E., Puchalt, J.C., Sánchez-Salmerón, A. "Improving skeleton algorithm for helping Caenorhabditis elegans trackers". Scientific Reports (2020) [doi](https://www.nature.com/articles/s41598-020-79430-8).
