# Chess_Policy_Value_Network_From_NumPy
Chess Policy and Value CNN form scratch in NumPy 


## 🧠 Network Overview

The network takes a chess position represented using **FEN (Forsyth-Edwards Notation)** and converts it into a tensor representation.

An Exmample Trained Model is included in The latest release
"Example_Model.npz"

The general pipeline is:

```text
FEN Position 
     │
     ▼
Board Tensor Encoding ()
     │
     ▼
Residual Convolutional Neural Network
     │
     ├───────────────┐
     ▼               ▼
 Policy Head      Value Head
     │               │
     ▼               ▼
Move Probabilities  Position Evaluation
```
Positional Evaluation is 20x8x8.
12 piece planes
6 game info planes (Castling Rights, En Passant Rights, Turn Plane (redundant))
2 attack planes -> showing squares us and our opponent are attacking. 

Move Encoding is AlphaZero style of 64x8x8
Each plane representes a direction and magnitude. E.g North 1 square is the top plane.
Network learns to map states -> starting_square,magnitude and direction. 


## 📁 Project Structure

```text
.
├── Example_Training_Data/
│
├── Dictionaries.pkl
│
├── FEN_2_TENSOR.py
├── RESIDUAL_CNN.py
├── USE_NETWORK.py
│
├──Example_Model
├── .gitignore
└── README.md
```

### `FEN_2_TENSOR.py`

Converts chess positions written in FEN notation into tensor representations that can be processed by the neural network.

### `RESIDUAL_CNN.py`

Meat of the project and contains an end to end training pipeline. Full NumPy implementation of batch Forward/Back prop as well as general training procedure.

### `USE_NETWORK.py`

Provides functionality for running chess positions through the trained network. The 'full_through' takes in a FEN string and returns the move scores, value from play to move perspective.

### `Dictinoaries.pkl` (various files)

Firstly, two others must be downloaded from latest release.
"move_TENS_dict.pkl"
"ID_TENS_dict.pkl"
These stores mappings used to translate between the neural network's move representation and chess moves.

### `Example_Training_Data/`

Contains example data used for training or demonstrating the network.
