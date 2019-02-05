# AlphaDraughts-Zero

This repository is the final project for *Reinforcement Learning - M2 MVA 2018*. Inspired by [AlphaGo Zero](https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html), we apply this method on [English Checkers](https://en.wikipedia.org/wiki/English_draughts), a famous strategy board games for two players. 


## Requirements

* Linux or macOS
* Python 3, version 3.4 or later is preferred
* PyTorch 1.0
* CPU or NVIDIA GPU + CUDA CuDNN

For pip users, run `pip install -r requirements.txt` to install dependencies.

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/Tong-ZHAO/AlphaDraughts-Zero
cd AlphaDraughts-Zero
pip install -r requirements.txt
```
### Train

- The training parameters should be specified in `./src/config.py` beforehand. <br>Some parameters can be passed to `./src/train.py` as arguments:

```
usage: train.py [-h] [--iterations N] [--lr LR] [--seed S] [--env ENV]

Training of AlphaDraughts Zero

optional arguments:
  -h, --help      show this help message and exit
  --iterations N  number of iterations of pipeline training)
  --lr LR         learning rate (default: 0.01)
  --seed S        random seed (default: 42)
  --env ENV       visdom environment
```

- To start the training:
```bash
cd src
python train.py
```
- To view loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To see more details of training, check the log file in `./logs/`.

<img src="/imgs/curves 2019-01-29 11_03.png" width="1200"/>

### Test

- To start the human-machine competition (qualitative evaluation):
```bash
cd src
python gui.py
```

- Some arguments could be passed to `./src/gui.py`:

```
usage: gui.py [-h] [--checkpoint C] [--human H] [--simulation S] [--ai A]

optional arguments:
  -h, --help      show this help message and exit
  --checkpoint C  which neural network model checkpoint to use.
  --human H       "white" or "black", which side human player plays, white
                  side always goes first.
  --simulation S  number of simulations for MCTS at each time step to choose
                  the action.
  --ai A          whether use AI, 1 means using AI, 0 means not using AI.
```

<img src="/imgs/GUI.png" width="400"/>

- Quantitative evaluation could be done using functions provided in `./src/elo.py`.

