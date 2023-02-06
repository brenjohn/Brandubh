# Brandubh
This repository has python code for playing a game called Brandubh and to use various reinforcement learning methods to train a bot to play against.

Brandubh is an Irish variation of the ancient board game known as Hnefatafl. The original rules of the game are unknown but a reconstruction of the rules based on boards found in Ireland and Irish poems detailing the game are discribed at the following link: 

https://www.ancientgames.org/hnefatafl-brandubh/

# Installing and playing Brandubh

First, create a new conda environment with the required dependencies installed:
```
conda create --name brandubh python=3.8
conda activate brandubh
conda install numpy tensorflow=2.4.1 keras=2.4.3
```

Then clone the repository:
```
git clone https://github.com/brenjohn/Brandubh.git
```

To download a trained bot to play against checkout the ZeroBot branch:
```
cd Brandubh
git checkout ZeroBot-15jan2023
```

To play the game run the play_brandubh script located in the top level directory:
```
python play_brandubh.py
```

# Running tests

To run unit tests use the following command from the top level directory:
```
python -m unittest
```

If the coverage package is installed, a test coverage report can be generated
with:
```
python -m coverage run -m unittest
python -m coverage html --omit=/tmp*
```
