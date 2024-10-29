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
    self.use_all_moves = True
    self.use_eval_normalisation = False
    self.norm_method = "standard"
    self.norm_factor = None
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
  
  def FEN_to_torch(self, fen_string, move=None):
    """
    Convert an FEN string into a torch tensor board representation
    """

    if move is None:
      boardvec = bf.FEN_to_board_vectors(fen_string)
    else:
      boardvec = bf.FEN_and_move_to_board_vectors(fen_string, move)

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

  def normalise_evaluations(self):
    """
    Normalise the evaluations to zero mean and unit variance, and save the scaling
    """
    if self.norm_method == "minmax":
      max_value = torch.max(-1 * torch.min(self.evals), torch.max(self.evals))
      self.norm_factor = max_value
      self.evals /= self.norm_factor
      if self.log_level > 0:
        print(f"Normalised evaluations, maximum value was {max_value}, now is {torch.max(-1 * torch.min(self.evals), torch.max(self.evals))}")
    
    elif self.norm_method == "standard":
      max_value = torch.max(-1 * torch.min(self.evals), torch.max(self.evals))
      mean = self.evals.mean()
      std = self.evals.std()
      self.evals = (self.evals - mean) / std
      new_max = torch.max(-1 * torch.min(self.evals), torch.max(self.evals))
      self.evals /= new_max
      self.norm_factor = (new_max, mean, std)
      if self.log_level > 0:
        print(f"Normalised evaluations, max_value = {max_value:.3f} (max used = {new_max:.3f}), mean = {mean.item():.3f}, std = {std.item():.3f}, now max value is {torch.max(-1 * torch.min(self.evals), torch.max(self.evals)):.3f}, mean is {self.evals.mean().item():.3f} and std is {self.evals.std().item():.3f}")

  def denomormalise_evaluation(self, value=None, all=False):
    """
    Convert a single value back to regular units (or do it for all saved values)
    """

    if self.norm_method == "minmax":
      if all:
        self.evals *= self.norm_factor
      elif value is not None:
        return value * self.norm_factor
      else:
        raise RuntimeError("EvalDataset.denormalise_evaluations() error: all=False and value=None, incorrect function inputs")
    
    if self.norm_method == "standard":
      if all:
        self.evals = (self.evals * self.norm_factor[0] * self.norm_factor[2]) + self.norm_factor[1]
      elif value is not None:
        return (value * self.norm_factor[0] * self.norm_factor[2]) + self.norm_factor[1]
      else:
        raise RuntimeError("EvalDataset.denormalise_evaluations() error: all=False and value=None, incorrect function inputs")

  def to_torch(self):
    """
    Convert dataset into torch tensors
    """

    if len(self.positions) == 0:
      print("EvalDataset.to_torch() warning: len(self.positions) == 0, nothing done")
      return

    # get the shape of the board tensors
    example = self.FEN_to_torch(self.positions[0].fen_string)
    num_pos = len(self.positions)

    if self.use_all_moves:
      # count how many positions we will have
      num_lines = 0
      for i in range(num_pos):
        num_lines += len(self.positions[i].move_vector)
      self.boards = torch.zeros((num_lines, *example.shape), dtype=example.dtype)
      self.evals = torch.zeros(num_lines, dtype=torch.float)
      add_ind = 0
      error_moves = 0
      if self.log_level > 0:
        print(f"self.use_all_moves = True, found {num_lines} lines (emerging from {num_pos} positions)")
    else:
      self.boards = torch.zeros((num_pos, *example.shape), dtype=example.dtype)
      self.evals = torch.zeros(num_pos, dtype=torch.float)
    
    for i in range(num_pos):

      if self.use_all_moves:
        # loop through all moves and add those boards
        for j in range(len(self.positions[i].move_vector)):
          if self.positions[i].move_vector[j].move_letters == "pv":
            error_moves += 1
            num_lines -= 1
            continue
          self.boards[add_ind] = self.FEN_to_torch(self.positions[i].fen_string,
                                                   self.positions[i].move_vector[j].move_letters)
          if self.positions[i].move_vector[j].eval == "mate":
            if not bf.is_white_next_FEN(self.positions[i].fen_string):
              self.evals[add_ind] = -self.mate_value
            else:
              self.evals[add_ind] = self.mate_value
          else:
            self.evals[add_ind] = self.positions[i].move_vector[j].eval
          if self.convert_evals_to_pawns:
            self.evals[add_ind] *= 1e-3
          add_ind += 1

      else:
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

    if self.use_all_moves:
      if error_moves > 0:
        self.boards = self.boards[:num_lines, :, :]
        self.evals = self.evals[:num_lines]
        if self.log_level > 0:
          print(f"The number of error moves was: {error_moves}, out of {num_lines + error_moves} lines. New vector length = {self.evals.shape}")

    # # for testing only
    # print("Shape of self.boards", self.boards.shape)
    # x = num_pos // 2
    # bf.print_FEN_board(self.positions[x].fen_string)
    # self.print_board_tensor(self.boards[x])

    if self.use_eval_normalisation:
      self.normalise_evaluations()

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

class ResidualBlock(nn.Module):

  def __init__(self, in_channels, out_channels, stride = 1, downsample = None):

    super(ResidualBlock, self).__init__()
    
    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
      nn.BatchNorm2d(out_channels)
    )

    self.downsample = downsample
    self.relu = nn.ReLU()
    self.out_channels = out_channels

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out
  
class ResNet(nn.Module):

  def __init__(self, block, layers, num_classes = 10):
    super(ResNet, self).__init__()
    self.inplanes = 64
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
      nn.BatchNorm2d(64),
      nn.ReLU()
    )
    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
    self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
    self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
    self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
    self.avgpool = nn.AvgPool2d(7, stride=1)
    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
          nn.BatchNorm2d(planes),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.layer0(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x

class ChessNet(nn.Module):

  def __init__(self, in_channels):

    super(ChessNet, self).__init__()

    self.in_channels = in_channels
    c_in = in_channels

    self.board_cnn = nn.Sequential(
      ResidualBlock(c_in, c_in),
      ResidualBlock(c_in, c_in),
      ResidualBlock(c_in, c_in),
      nn.Sequential(nn.Flatten(), nn.Linear(c_in * 8 * 8, c_in), nn.ReLU()),
      nn.Linear(c_in, 1),
    )

  def forward(self, x):
    for l in self.layers:
      x = l(x)
    return x

# ----- old, not currently used ----- #

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
    
# ----- training functionality ----- #
  
def train(net, data_x, data_y, epochs=1, lr=5e-5, device="cuda"):
  """
  Perform a training epoch for a given network based on data inputs
  data_x, and correct outputs data_y
  """

  # move onto the specified device
  net.board_cnn.to(device)
  data_x = data_x.to(device)
  data_y = data_y.to(device)

  # put the model in training mode
  net.board_cnn.train()

  lossfcn = nn.MSELoss()
  optim = torch.optim.Adam(net.board_cnn.parameters(), lr=lr)

  batch_size = 64
  num_batches = len(data_x) // batch_size

  for i in range(epochs):

    print(f"Starting epoch {i + 1}. There will be {num_batches} batches")

    rand_idx = torch.randperm(data_x.shape[0])
    avg_loss = 0

    for n in range(num_batches):

      batch_x = data_x[rand_idx[n * batch_size : (n+1) * batch_size]]
      batch_y = data_y[rand_idx[n * batch_size : (n+1) * batch_size]]

      # use the model for a prediction and calculate loss
      net_y = net.board_cnn(batch_x)
      loss = lossfcn(net_y.squeeze(1), batch_y)

      # backpropagate
      loss.backward()
      optim.step()
      optim.zero_grad()

      avg_loss += loss.item()

      # if n % 500 == 0:
      #   print(f"Loss is {(avg_loss / (n + 1)) * 1000:.3f}, epoch {i + 1}, batch {n + 1} / {num_batches}")
    
    print(f"Loss is {(avg_loss / (num_batches * batch_size)) * 1000:.3f}, at end of epoch {i + 1}")

  return net

def normalise_data(data, factors):
  """
  Normalise data based on [max, mean, std]
  """
  max, mean, std = factors
  return ((data - mean) / std) / max

def denormalise_data(data, factors):
  """
  Undo normalisation based on [max, mean, std]
  """
  max, mean, std = factors
  return (data * max * std) + mean

def train_procedure(net, dataname, dataloader, data_inds, norm_factors,
                    epochs=1, lr=1e-7, device="cuda", batch_size=64,
                    loss_style="MSE"):
  """
  Perform a training epoch for a given network based on data inputs
  data_x, and correct outputs data_y
  """

  # move onto the specified device
  net.board_cnn.to(device)

  # put the model in training mode
  net.board_cnn.train()

  if loss_style.lower() == "mse":
    lossfcn = nn.MSELoss()
  elif loss_style.lower() == "l1":
    lossfcn = nn.L1Loss()
  elif loss_style.lower() == "huber":
    lossfcn == nn.HuberLoss()
  else:
    raise RuntimeError(f"train_procedure() error: loss_style = {loss_style} not recognised")

  optim = torch.optim.Adam(net.board_cnn.parameters(), lr=lr)
  
  # each epoch, cover the entire training dataset
  for i in range(epochs):

    print(f"Starting epoch {i + 1}.")
    total_batches = 0
    epoch_loss = 0

    # load the dataset in a series of slices
    for slice_num, j in enumerate(data_inds):

      # load this segment of the dataset
      dataset = dataloader.load(dataname, id=j)
      data_x = dataset.boards
      data_y = dataset.evals

      # normalise y labels
      data_y = normalise_data(data_y, norm_factors)

      num_batches = len(data_x) // batch_size
      total_batches += num_batches
      rand_idx = torch.randperm(data_x.shape[0])
      avg_loss = 0

      print(f"Starting slice {slice_num + 1} / {len(data_inds)}. There will be {num_batches} batches. ", end="", flush=True)

      # iterate through each batch for this slice of the dataset
      for n in range(num_batches):

        batch_x = data_x[rand_idx[n * batch_size : (n+1) * batch_size]]
        batch_y = data_y[rand_idx[n * batch_size : (n+1) * batch_size]]

        # go to cuda
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # use the model for a prediction and calculate loss
        net_y = net.board_cnn(batch_x)
        loss = lossfcn(net_y.squeeze(1), batch_y)

        # backpropagate
        loss.backward()
        optim.step()
        optim.zero_grad()

        avg_loss += loss.item()

        # if n % 500 == 0:
        #   print(f"Loss is {(avg_loss / (n + 1)) * 1000:.3f}, epoch {i + 1}, batch {n + 1} / {num_batches}")

      # this dataset slice is finished
      epoch_loss += avg_loss
      avg_loss = avg_loss / num_batches
      avg_loss = avg_loss ** 0.5 * norm_factors[0] * norm_factors[2] # try to scale to original units
      print(f"Loss is {avg_loss:.3f}, during epoch {i + 1}, slice {slice_num + 1} / {len(data_inds)}", flush=True)
  
    # this epoch is finished
    epoch_loss = epoch_loss / total_batches
    epoch_loss = epoch_loss ** 0.5 * norm_factors[0] * norm_factors[2] # try to scale to original units
    print(f"Epoch {i + 1} has finished after {total_batches} batches. Overall average loss = {epoch_loss:.3f}", flush=True)

  # finally, return the network that we have trained
  return net

if __name__ == "__main__":

  t1 = time.time()

  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=10)                   # number of training epochs
  parser.add_argument("-lr", "--learning-rate", type=float, default=1e-7)   # learning rate during training
  parser.add_argument("--device", default="cuda")                         # device to use, "cpu" or "cuda"
  parser.add_argument("--batch-size", type=int, default=64)               # size of learning batches
  parser.add_argument("--loss-style", default="MSE")                      # loss function, 'MSE', 'L1', or 'Huber'

  args = parser.parse_args()

  # create the network based on 19 layer boards
  net = ChessNet(19)

  # initiate the training procedure
  trained_net = train_procedure(
    net=net,
    dataname="datasetv1",
    dataloader=ModelSaver("/home/luke/chess/python/datasets/", log_level=1),
    data_inds=list(range(1, 11)),
    norm_factors=[23.927, -0.240, 0.355],
    epochs=args.epochs,
    lr=args.learning_rate,
    device=args.device,
    batch_size=args.batch_size,
    loss_style=args.loss_style,  
  )

  # save the finished model
  modelsaver = ModelSaver("/home/luke/chess/python/models/")
  modelsaver.save("chessnet_model", trained_net)

  t2 = time.time()

  print(f"Finished training network, after {(t2 - t1) / 3600:.2f} hours")