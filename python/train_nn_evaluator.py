import modules.board_module as bf
import modules.tree_module as tf
import modules.stockfish_module as sf
from ModelSaver import ModelSaver
import random
from dataclasses import dataclass
from collections import namedtuple
import itertools
import time
import argparse
import numpy as np
from math import floor, ceil

import torch
import torch.nn as nn

from assemble_data import Move, Position

# Move = namedtuple("Move",
#                   ("move_letters", "eval", "depth", "ranking"))
# Position = namedtuple("Position",
#                       ("fen_string", "eval", "move_vector"))

class EvalDataset(torch.utils.data.Dataset):

  def __init__(self, datapath, sample_names, indexes=None, log_level=1):
    """
    Dataset containint stockfish evaluations of chess positions. Pass in the
    path to the samples, their names, and the indexes to load
    """

    t1 = time.time()

    self.modelsaver = ModelSaver(datapath, log_level=log_level)
    self.log_level = log_level
    self.positions = []
    self.mate_value = 50000 # 50 pawns
    self.convert_evals_to_pawns = True
    self.board_dtype = torch.float

    self.boards = []
    self.evals = []

    # automatically get all indexes if not specified
    if indexes is None:
      indexes = list(range(self.modelsaver.get_recent_file(name=sample_names, 
                                                           return_int=True) + 1))

    for ind in indexes:
      newdata = self.modelsaver.load(sample_names, id=ind)
      self.positions += newdata

    t2 = time.time()

    if self.log_level >= 1:
      print(f"EvalDataset(): {len(indexes)} files loaded {t2 - t1:.2f} seconds")

  def __len__(self):
    return len(self.positions)
  
  def __getitem__(self, idx):
    if idx > len(self.positions):
      raise RuntimeError(f"EvalDataset.__getitem__() error: idx ({idx}) > number of samples (len{self.positions})")
    
    return self.positions[idx]
  
  def FEN_to_torch(self, fen_string):
    """
    Convert an FEN string into a torch tensor board representation
    """

    boardvec = bf.FEN_to_board_vectors(fen_string)
    tensortype = self.board_dtype

    t_wP = torch.tensor(boardvec.wP, dtype=tensortype).reshape(8, 8)
    t_wN = torch.tensor(boardvec.wN, dtype=tensortype).reshape(8, 8)
    t_wB = torch.tensor(boardvec.wB, dtype=tensortype).reshape(8, 8)
    t_wR = torch.tensor(boardvec.wR, dtype=tensortype).reshape(8, 8)
    t_wQ = torch.tensor(boardvec.wQ, dtype=tensortype).reshape(8, 8)
    t_wK = torch.tensor(boardvec.wK, dtype=tensortype).reshape(8, 8)
    t_bP = torch.tensor(boardvec.bP, dtype=tensortype).reshape(8, 8)
    t_bN = torch.tensor(boardvec.bN, dtype=tensortype).reshape(8, 8)
    t_bB = torch.tensor(boardvec.bB, dtype=tensortype).reshape(8, 8)
    t_bR = torch.tensor(boardvec.bR, dtype=tensortype).reshape(8, 8)
    t_bQ = torch.tensor(boardvec.bQ, dtype=tensortype).reshape(8, 8)
    t_bK = torch.tensor(boardvec.bK, dtype=tensortype).reshape(8, 8)
    t_wKS = torch.tensor(boardvec.wKS, dtype=tensortype).reshape(8, 8)
    t_wQS = torch.tensor(boardvec.wQS, dtype=tensortype).reshape(8, 8)
    t_bKS = torch.tensor(boardvec.bKS, dtype=tensortype).reshape(8, 8)
    t_bQS = torch.tensor(boardvec.bQS, dtype=tensortype).reshape(8, 8)
    t_colour = torch.tensor(boardvec.colour, dtype=tensortype).reshape(8, 8)
    t_total_moves = torch.tensor(boardvec.total_moves, dtype=tensortype).reshape(8, 8)
    t_no_take_ply = torch.tensor(boardvec.no_take_ply, dtype=tensortype).reshape(8, 8)

    board_tensor = torch.stack((
      t_wP,
      t_wN,
      t_wB,
      t_wR,
      t_wQ,
      t_wK,
      t_bP,
      t_bN,
      t_bB,
      t_bR,
      t_bQ,
      t_bK,
      t_wKS,
      t_wQS,
      t_bKS,
      t_bQS,
      t_colour,
      t_total_moves,
      t_no_take_ply,
    ), dim=0)

    return board_tensor
  
  def print_board_tensor(self, board_tensor):
    """
    Print the elements of the board tensor
    """

    print("Board tensor shape:", board_tensor.shape)
    print("White pawns", board_tensor[0])
    print("White knights", board_tensor[1])
    print("White bishops", board_tensor[2])
    print("White rooks", board_tensor[3])
    print("White queen", board_tensor[4])
    print("White king", board_tensor[5])
    print("Black pawns", board_tensor[6])
    print("Black knights", board_tensor[7])
    print("Black bishops", board_tensor[8])
    print("Black rooks", board_tensor[9])
    print("Black queen", board_tensor[10])
    print("Black king", board_tensor[11])
    print("White castles KS", board_tensor[12])
    print("White castles QS", board_tensor[13])
    print("Black castles KS", board_tensor[14])
    print("Black castles QS", board_tensor[15])
    print("colour", board_tensor[16])
    print("total moves", board_tensor[17])
    print("no take ply", board_tensor[18])

    return
  
  def to_torch(self):
    """
    Convert dataset into torch tensors
    """

    if len(self.positions) == 0:
      print("EvalDataset.to_torch() warning: len(self.positions) == 0, nothing done")
      return

    board_tensors = []

    # get the shape of the board tensors
    example = self.FEN_to_torch(self.positions[0].fen_string)
    num_pos = len(self.positions)

    self.boards = torch.zeros((num_pos, *example.shape), dtype=example.dtype)
    self.evals = torch.zeros(num_pos, dtype=torch.float)
    
    for i in range(len(self.positions)):
      self.boards[i] = self.FEN_to_torch(self.positions[i].fen_string)
      if self.positions[i].eval == "mate":
        if bf.is_white_next_FEN(self.positions[i].fen_string):
          self.evals[i] = -self.mate_value
        else:
          self.evals[i] = self.mate_value
      else:
        self.evals[i] = self.positions[i].eval
      if self.convert_evals_to_pawns:
        self.evals[i] *= 1e-3

    # # for testing only
    # print("Shape of self.boards", self.boards.shape)
    # x = num_pos // 2
    # bf.print_FEN_board(self.positions[x].fen_string)
    # self.print_board_tensor(self.boards[x])

    return
  
  def check_duplicates(self, remove=False):
    """
    Check the number (and potentially remove) duplicates
    """

    t1 = time.time()

    # remove duplicates
    seen_values = set()
    unique_positions = []

    for position in self.positions:
      if position.fen_string not in seen_values:
        seen_values.add(position.fen_string)
        unique_positions.append(position)

    num_duplicates = len(self.positions) - len(unique_positions)

    # now if we want remove the duplicates
    if remove: self.positions = unique_positions

    t2 = time.time()

    if self.log_level >= 1:
      print(f"EvalDataset(): {num_duplicates} duplicates found in {t2 - t1:.2f} seconds{', and removed' if remove else ''}")

    return num_duplicates
  
  def check_mate_positions(self, remove=False):
    """
    Check the number (and potentially remove) mate positions
    """

    t1 = time.time()

    no_mate_positions = []

    # loop backwards over all positions checking for mate
    for i in range(len(self.positions) - 1, -1, -1):
      if self.positions[i].eval != "mate":
        no_mate_positions.append(self.positions[i])

    num_mates = len(self.positions) - len(no_mate_positions)

    # now if we want remove the duplicates
    if remove: self.positions = no_mate_positions

    t2 = time.time()

    if self.log_level >= 1:
      print(f"EvalDataset(): {num_mates} mate positions found in {t2 - t1:.2f} seconds{', and removed' if remove else ''}")

    return num_mates

def calc_conv_layer_size(W, H, C, kernel_num, kernel_size, stride, padding, prnt=False):

  new_W = floor(((W - kernel_size + 2*padding) / (stride)) + 1)
  new_H = floor(((H - kernel_size + 2*padding) / (stride)) + 1)

  if prnt:
    print(f"Convolution transforms ({C}x{W}x{H}) to ({kernel_num}x{new_W}x{new_H})")

  return new_W, new_H, kernel_num

def calc_max_pool_size(W, H, C, pool_size, stride, prnt=False):

  new_W = floor(((W - pool_size) / stride) + 1)
  new_H = floor(((H - pool_size) / stride) + 1)

  if prnt:
    print(f"Max pool transforms ({C}x{W}x{H}) to ({C}x{new_W}x{new_H})")

  return new_W, new_H, C

def calc_adaptive_avg_size(W, H, C, output_size, prnt=False):

  if prnt:
    print(f"Adaptive pool transforms ({C}x{W}x{H}) to ({C}x{output_size[0]}x{output_size[1]})")

  return output_size[0], output_size[1], C

def calc_FC_layer_size(W, H, C, prnt=False):

  new_W = 1
  new_H = 1
  new_C = W * H * C

  if prnt:
    print(f"The first FC layer should take size ({C}x{W}x{H}) as ({new_C}x{new_W}x{new_H})")

  return new_W, new_H, new_C

class BoardCNN(nn.Module):

  name = "BoardCNN"

  def __init__(self, board_size, outputs, device):

    super(BoardCNN, self).__init__()
    self.device = device

    (channel, width, height) = board_size
    self.name += f"_{channel}x{width}x{height}"

    # # calculate the size of the first fully connected layer (ensure settings match image_features_ below)
    # w, h, c = calc_conv_layer_size(width, height, channel, kernel_num=16, kernel_size=5, stride=2, padding=2, print=False)
    # w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, print=False)
    # w, h, c = calc_conv_layer_size(w, h, c, kernel_num=64, kernel_size=5, stride=2, padding=2, print=False)
    # w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, print=False)
    # w, h, c = calc_FC_layer_size(w, h, c, print=False)
    # fc_layer_num = c

    # # define the CNN to handle the images
    # self.board_cnn = nn.Sequential(

    #   # input CxWxH, output 2CxWxH
    #   nn.Conv2d(channel, channel * 2, kernel_size=5, stride=2, padding=2),
    #   nn.ReLU(),
    #   nn.MaxPool2d(kernel_size=3, stride=2),
    #   # nn.Dropout(),
    #   nn.Conv2d(channel * 2, channel * 4, kernel_size=5, stride=2, padding=2),
    #   nn.ReLU(),
    #   nn.MaxPool2d(kernel_size=3, stride=2),
    #   # nn.Dropout(),
    #   nn.Flatten(),
    #   nn.Linear(fc_layer_num, 128),
    #   nn.ReLU(),
    #   # nn.Linear(64*16, 64),
    #   # nn.ReLU(),
    # )

    self.board_cnn = nn.Sequential(

      # Layer 1
      nn.Conv2d(in_channels=19, out_channels=32, kernel_size=3, padding=1),  # Conv layer
      nn.ReLU(),                                                             # Activation
      nn.MaxPool2d(kernel_size=2),                                           # Pooling (output size: 32 x 4 x 4)

      # Layer 2
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),                                           # Pooling (output size: 64 x 2 x 2)

      # Layer 3
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),                                           # Pooling (output size: 128 x 1 x 1)

      # Flatten layer to transition to fully connected
      nn.Flatten(),

      # two fully connected layers to produce a single output
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
  )

  def forward(self, board):
    board = board.to(self.device)
    x = self.board_cnn(board)
    return x

  def to_device(self, device):
    self.board_cnn.to(device)
    
  
def train(net, data_x, data_y, epochs=1, lr=5e-5, device="cuda"):
  """
  Perform a training epoch for a given network based on data inputs
  data_x, and correct outputs data_y
  """

  # move onto the specified device
  net.board_cnn.to(device)
  data_x.to(device)
  data_y.to(device)

  # put the model in training mode
  net.board_cnn.train()

  loss = nn.MSELoss()
  optim = torch.optim.Adam(net.board_cnn.parameters(), lr=lr)

  rand_idx = torch.randperm(data_x.shape[0])
  batch_size = 32
  num_batches = len(data_x) // batch_size

  for i in range(epochs):

    print(f"Starting epoch {i + 1}. There will be {num_batches} batches")

    for n in num_batches:

      batch_x = data_x[rand_idx[n * batch_size : (n+1) * batch_size]]
      batch_y = data_y[rand_idx[n * batch_size : (n+1) * batch_size]]

      # use the model for a prediction and calculate loss
      net_y = net(batch_x)
      loss = loss(net_y, data_y)

      # backpropagate
      loss.backward()
      optim.step()
      optim.zero_grad()

      if n % 10 == 0:
        print(f"Loss is {loss.item():.3f}, epoch {i + 1}, batch {n} / {num_batches}")

  return net

if __name__ == "__main__":

  num_rand = 4096
  datapath = "/home/luke/chess/python/gamedata/samples"
  eval_file_template = "random_n={0}_sample"
  inds = [1]
  dataset = EvalDataset(datapath, eval_file_template.format(num_rand),
                        indexes=inds)
  
  t = dataset.FEN_to_torch(dataset.positions[10].fen_string)

  dataset.check_mate_positions(remove=True)
  dataset.to_torch()
  
  # num_duplicates = dataset.check_duplicates(remove=True)
  # num_mates = dataset.check_mate_positions(remove=True)

  # num_duplicates = dataset.check_duplicates()
  # num_mates = dataset.check_mate_positions()