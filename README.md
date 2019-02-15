# Object Recognition
One-Shot object recognition thesis project by Alexander Holstrup and Cristian Botezatu

## Introduction
Object recognition is the subfield of computer vision that aims to solve the problem of detecting one or more objects in an image, providing not only the class of the objects but the position of them in that coordinate frame. 
The objective of this project would be to create a model that can perform one shot object recognition. I.e. Given an object recognition dataset with n classes C we aim to create a model that given a single image of class Cn+1, which the model has never seen before, it would be able to find this class on new samples. However, as a start, we will allow our model to get trained as much as needed, so that the accuracy could be increased. Afterwards, by maintaining the high accuracy, the number of trainings will be reduced until, eventually, reaching the one-shot object recognition.


## Setup

Developed in Python 2

### Install Libraries Needed
```
pip install -r  requirements.txt
```


## References


## Acknowledgements
Thank you to [Uizard Technologies](https://uizard.io/) for guidance through the project and to professor [Ole Winther](https://www.dtu.dk/english/service/phonebook/person?id=10167&cpid=109334&tab=3&qt=dtuprojectquery#tabs) from [Technical University of Denmark](https://www.dtu.dk).
