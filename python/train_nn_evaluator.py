import modules.board_module as bf
import modules.tree_module as tf
import modules.stockfish_module as sf
from ModelSaver import ModelSaver
import os
import random
from dataclasses import dataclass, asdict
from collections import namedtuple
import itertools
import time
import argparse
import numpy as np
from math import floor, ceil
from datetime import datetime

import torch
import torch.nn as nn

# Move = namedtuple("Move",
#                   ("move_letters", "eval", "depth", "ranking"))
# Position = namedtuple("Position",
#                       ("fen_string", "eval", "move_vector"))

class EvalDataset(torch.utils.data.Dataset):

  def __init__(self, datapath, sample_names, auto_load=True, indexes=None, log_level=1):
    """
    Dataset containint stockfish evaluations of chess positions. Pass in the
    path to the samples, their names, and the indexes to load
    """

    t1 = time.time()

    self.log_level = log_level
    self.positions = []
    self.seen_values = None
    self.num_lines = None

    self.mate_value = 50                # value of a checkmate (units: 1.0 per pawn)
    self.convert_evals_to_pawns = True  # convert evaluations from 1000 per pawn, to 1.0 per pawn
    self.use_all_moves = True           # add in all child moves from a positions, instead of the parent
    self.save_sq_eval = True            # evaluate every square in the board using my handcrafted evaluator
    self.use_eval_normalisation = False # apply normalisation to all evaluations
    self.norm_method = "standard"       # normalisation method to use, standard is mean/std scale -1/+1 bound
    self.norm_factor = None             # normalisation factors saved for future use
    self.board_dtype = torch.float      # datatype to use for torch tensors
    self.stockfish_converstion = 1e-2   # stockfish evals are in 1/100th of a pawn, so 200 = 2 pawns
    self.my_conversion = 1e-3           # my evals are in 1/1000th of a pawn, so 2000 = 2 pawns

    # create empty containers for future data
    self.boards = []
    self.evals = []
    self.square_evals = []

    if auto_load:

      self.modelsaver = ModelSaver(datapath, log_level=log_level)

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
  
  def FEN_to_torch(self, fen_string, move=None, eval_sqs=False):
    """
    Convert an FEN string into a torch tensor board representation
    """

    if move is None:
      if eval_sqs:
        boardvec = bf.FEN_to_board_vectors_with_eval(fen_string)
      else:
        boardvec = bf.FEN_to_board_vectors(fen_string)
    else:
      if eval_sqs:
        boardvec = bf.FEN_move_eval_to_board_vectors(fen_string, move)
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

    # ignore these as no data in them currently, just wasted space
    # t_total_moves = torch.tensor(boardvec.total_moves, dtype=tensortype).reshape(8, 8)
    # t_no_take_ply = torch.tensor(boardvec.no_take_ply, dtype=tensortype).reshape(8, 8)

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
      # t_total_moves,
      # t_no_take_ply,
    ), dim=0)

    if eval_sqs:
      sq_evals = torch.tensor(boardvec.sq_evals, dtype=float)
      if self.convert_evals_to_pawns:
        sq_evals *= self.my_conversion
      return board_tensor, sq_evals
    else:
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
    # print("total moves", board_tensor[17])
    # print("no take ply", board_tensor[18])

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

  def count_all_positions(self):
    """
    Count the total number of possible positions
    """

    num_pos = len(self.positions)

    if not self.use_all_moves: return num_pos

    num_lines = 0

    # loop through all positions and all child moves
    for i in range(num_pos):
      for j in range(len(self.positions[i].move_vector)):

        if self.positions[i].move_vector[j].move_letters == "pv":
          continue

        num_lines += 1

    self.num_lines = num_lines

    return num_lines

  def to_torch(self, indexes_only=None):
    """
    Convert dataset into torch tensors
    """

    t1 = time.time()

    if len(self.positions) == 0:
      print("EvalDataset.to_torch() warning: len(self.positions) == 0, nothing done")
      return

    # get the shape of the board tensors
    example = self.FEN_to_torch(self.positions[0].fen_string)
    num_pos = len(self.positions)

    break_out = False

    if indexes_only is not None:
      indexes_only = sorted(indexes_only)
      proxy_ind = 0
      selected_indexes_ind = 0

    if self.use_all_moves:
      # count how many positions we will have
      num_lines = 0
      for i in range(num_pos):
        num_lines += len(self.positions[i].move_vector)
      if indexes_only is not None:
        if len(indexes_only) > num_lines:
          raise RuntimeError(f"EvalDataset.to_torch() error: num_lines = {num_lines}, but number of selected indexes exceeds this = {len(indexes_only)}")
        num_lines = len(indexes_only)
      self.boards = torch.zeros((num_lines, *example.shape), dtype=example.dtype)
      self.evals = torch.zeros(num_lines, dtype=torch.float)
      if self.save_sq_eval:
        self.square_evals = torch.zeros((num_lines, 64), dtype=torch.float)
      error_moves = 0
      if self.log_level > 0:
        if indexes_only is not None:
          print(f"self.use_all_moves = True, selected {num_lines} lines using indexes_only (total {num_pos} positions)")
        else:
          print(f"self.use_all_moves = True, found {num_lines} lines (emerging from {num_pos} positions)")
    else:
      if indexes_only is not None:
        if len(indexes_only) > num_pos:
          raise RuntimeError(f"EvalDataset.to_torch() error: num_pos = {num_pos}, but number of selected indexes exceeds this = {len(indexes_only)}")
        num_gen = len(indexes_only)
      else: num_gen = num_pos
      self.boards = torch.zeros((num_gen, *example.shape), dtype=example.dtype)
      self.evals = torch.zeros(num_gen, dtype=torch.float)
      if self.save_sq_eval:
        self.square_evals = torch.zeros((num_gen, 64), dtype=torch.float)

    add_ind = 0
    
    for i in range(num_pos):

      if break_out: break

      if self.use_all_moves:
        # loop through all moves and add those boards
        for j in range(len(self.positions[i].move_vector)):

          if break_out: break

          if self.positions[i].move_vector[j].move_letters == "pv":
            error_moves += 1
            if indexes_only is None:
              num_lines -= 1
            continue

          if indexes_only is not None:
            # is this index one we want to include
            if indexes_only[selected_indexes_ind] == proxy_ind:
              selected_indexes_ind += 1
              # check if we have finished this batch
              if selected_indexes_ind >= len(indexes_only):
                break_out = True
                break
            else:
              # skip the current index
              proxy_ind += 1
              continue

          if self.save_sq_eval:
            self.boards[add_ind], self.square_evals[add_ind] = self.FEN_to_torch(
              self.positions[i].fen_string, self.positions[i].move_vector[j].move_letters, self.save_sq_eval
            )
          else:
            self.boards[add_ind] = self.FEN_to_torch(self.positions[i].fen_string,
                                                    self.positions[i].move_vector[j].move_letters)
          # who is next from the PARENT position
          white_next = bf.is_white_next_FEN(self.positions[i].fen_string)
          if self.positions[i].move_vector[j].eval == "mate":
            if not white_next:
              self.evals[add_ind] = -self.mate_value / self.stockfish_converstion
            else:
              self.evals[add_ind] = self.mate_value / self.stockfish_converstion
          else:
            sign = (white_next * 2) - 1
            self.evals[add_ind] = self.positions[i].move_vector[j].eval * sign # if white next at parent, engine is white for child
          if self.convert_evals_to_pawns:
            self.evals[add_ind] *= self.stockfish_converstion

          add_ind += 1

      else:

        if indexes_only is not None:
          # is this index one we want to include
          if indexes_only[selected_indexes_ind] == proxy_ind:
            selected_indexes_ind += 1
            # check if we have finished this batch
            if selected_indexes_ind >= len(indexes_only):
              break_out = True
              break
          else:
            # skip the current index
            proxy_ind += 1
            continue

        if self.save_sq_eval:
          self.boards[add_ind], self.square_evals[add_ind] = self.FEN_to_torch(
            self.positions[i].fen_string, eval_sqs=self.save_sq_eval
          )
        else:
          self.boards[add_ind] = self.FEN_to_torch(self.positions[i].fen_string,
                                                  self.positions[i].move_vector[j].move_letters)
        white_next = bf.is_white_next_FEN(self.positions[i].fen_string)
        sign = (white_next * 2) - 1

        # if white plays next, a + evaluation is appropriate
        # if black plays next, a + evaluation means good for black, so should be flipped

        sign = 1 # DISABLE THIS SWITCHING

        if self.positions[add_ind].eval == "mate":
          if bf.is_white_next_FEN(self.positions[i].fen_string):
            self.evals[add_ind] = -self.mate_value / self.stockfish_converstion
          else:
            self.evals[add_ind] = self.mate_value / self.stockfish_converstion
        else:
          self.evals[add_ind] = int(self.positions[add_ind].eval) * sign # convert to OBJECTIVE evaluation
        if self.convert_evals_to_pawns:
          self.evals[add_ind] *= self.stockfish_converstion

        add_ind += 1

    if self.use_all_moves and indexes_only is None:
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
      if self.log_level > 0:
        print("EvalDataset() is applying normalisation to self.evals")
      self.normalise_evaluations()

    t2 = time.time()

    if self.log_level > 0:
      if self.use_all_moves:
        total_num = num_lines
      else: total_num = num_gen
      print(f"EvalDataset(): {total_num} positions converted in {t2 - t1:.2f} seconds, average {((t2 - t1) / total_num) * 1e3:.3f} ms per position")

    return
  
  def check_duplicates(self, remove=False, wipe_seen=True):
    """
    Check the number (and potentially remove) duplicates
    """

    t1 = time.time()

    # remove duplicates
    if wipe_seen:
      self.seen_values = set()
    elif self.seen_values is None:
      self.seen_values = set()
      
    unique_positions = []

    for position in self.positions:
      if position.fen_string not in self.seen_values:
        self.seen_values.add(position.fen_string)
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

  def __init__(self, in_channels, out_channels, num_blocks=3):

    super(ChessNet, self).__init__()

    self.in_channels = in_channels
    c_in = in_channels

    blocks = [ResidualBlock(c_in, c_in) for i in range(num_blocks)]

    self.board_cnn = nn.Sequential(
      *blocks,
      nn.Sequential(nn.Flatten(), nn.Linear(c_in * 8 * 8, c_in), nn.ReLU()),
      nn.Linear(c_in, out_channels),
    )

  def forward(self, x):
    for l in self.board_cnn:
      x = l(x)
    return x

class FCResidualBlock(nn.Module):

  def __init__(self, in_channels, numlayers=1, dropout_prob=0.0):

    super(FCResidualBlock, self).__init__()

    block = [nn.Sequential(
      nn.Linear(in_channels, in_channels),
      nn.ReLU(),
      nn.Dropout(p=dropout_prob),
    ) for i in range(numlayers)]

    self.net = nn.Sequential(*block)
    self.relu = nn.ReLU()

  def forward(self, x):
    residual = x
    out = self.net(x)
    out = out + residual
    out = self.relu(out)
    return out

class FCChessNet(nn.Module):

  def __init__(self, in_channels, out_channels, num_blocks=3, dropout_prob=0.0,
               block_type="residual", layer_scaling=64):

    super(FCChessNet, self).__init__()

    self.in_channels = in_channels

    if block_type == "residual":
      blocks = [FCResidualBlock(in_channels * layer_scaling, 
                                dropout_prob=dropout_prob,
                                numlayers=3) for i in range(num_blocks)]
    
    elif block_type == "simple":
      blocks = [nn.Sequential(
        nn.Linear(in_channels * layer_scaling, in_channels * layer_scaling), 
        nn.ReLU(),
        nn.Dropout(p=dropout_prob),
      ) for i in range(num_blocks)]
    else:
      raise RuntimeError(f"FCChessNet.__init__() error: block_type={block_type} not recognised")

    self.board_cnn = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_channels * 64, in_channels * layer_scaling),
      nn.ReLU(),
      *blocks,
      nn.Linear(in_channels * layer_scaling, out_channels),
    )

  def forward(self, x):
    for l in self.board_cnn:
      x = l(x)
    return x
  
class ChessNetAttention(nn.Module):

  def __init__(self, in_channels, penultimate_channels=64, num_blocks=1, layer_scaling=64, dropout_prob=0):

    super(ChessNetAttention, self).__init__()

    blocks = [nn.Sequential(
      nn.Linear(in_channels * layer_scaling, in_channels * layer_scaling), 
      nn.ReLU(),
      nn.Dropout(p=dropout_prob),
    ) for i in range(num_blocks)]

    self.board_cnn = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_channels * 64, in_channels * layer_scaling),
      nn.ReLU(),
      *blocks,
      nn.Linear(in_channels * layer_scaling, penultimate_channels),
    )

    # penultimate layer with 64 outputs (square evaluations)
    self.square_eval_layer = nn.Linear(penultimate_channels, 64)
    
    # attention layer to compute attention weights for each square
    self.attention_layer = nn.Linear(penultimate_channels, 64)  # Produces 64 attention weights

    # final layer that takes the attention-weighted sum of square evaluations to produce a board evaluation
    self.board_eval_layer = nn.Linear(64, 1)
  
  def forward(self, x):

    # go through main network
    for l in self.board_cnn:
      x = l(x)
    
    # now generate square-by-square evaluations
    square_eval = self.square_eval_layer(x)  # Output shape: [batch_size, 64]
    
    # compute attention weights
    attention_weights = nn.functional.softmax(self.attention_layer(x), dim=1)  # Output shape: [batch_size, 64]
    
    # apply attention weights to square evaluations
    attention_applied = square_eval * attention_weights  # Element-wise multiplication; shape: [batch_size, 64]
    
    # aggregate weighted square evaluations into a single board evaluation
    board_eval = self.board_eval_layer(attention_applied)

    # combine into a single tensor [batch, 65 (elem0=board eval, elem1-64=square evals)]
    output = torch.cat((board_eval, square_eval), dim=1)
    
    return output
    
# ----- training functionality ----- #
  
def vary_all_inputs(raw_inputarg=None, param_1=None, param_2=None, param_3=None, repeats=None):
  """
  Helper function for adjusting parameters. With param_1 set to list_1 and param_2 set to list_2:

  The pattern goes (with param_1=[A,B,C...] and param_2=[1,2,3...])
    A1, A2, A3, ...
    B1, B2, B3, ...
    C1, C2, C3, ...

  With param_3=[X,Y,Z,...] we repeat the above grid first for X, then Y etc

  Set repeats to get sequential repeats, eg repeats=3 gives
    A1, A1, A1, A2, A2, A2, A3, A3, A3, ...
  """

  # convert input arg from 1...Max to 0...Max-1
  inputarg = raw_inputarg - 1

  # understand inputs
  if param_1 is not None:
    if isinstance(param_1, list):
      list_1 = param_1
    else:
      list_1 = [param_1]
    len_list_1 = len(list_1)
  else: return None, None, None

  if param_2 is not None:
    if isinstance(param_2, list):
      list_2 = param_2
    else:
      list_2 = [param_2]
    len_list_2 = len(list_2)
  else:
    len_list_2 = 1

  if param_3 is not None:
    if param_2 is None: raise RuntimeError("param_2 must be specified before param_3 in vary_all_inputs()")
    if isinstance(param_3, list):
      list_3 = param_3
    else:
      list_3 = [param_3]
    len_list_3 = len(list_3)
  else:
    len_list_3 = 1

  if repeats is None: repeats = 1

  # how fast do we move through lists
  list_1_changes = repeats
  list_2_changes = repeats * len_list_1
  list_3_changes = repeats * len_list_1 * len_list_2

  # don't allow overflow
  num_trainings = len_list_1 * len_list_2 * len_list_3 * repeats
  if raw_inputarg > num_trainings:
    raise RuntimeError(f"vary_all_inputs() got raw_inputarg={raw_inputarg} too high, num_trainings={num_trainings}")

  var_1 = list_1[(inputarg // list_1_changes) % len_list_1]
  if param_2 is not None:
    var_2 = list_2[(inputarg // list_2_changes) % len_list_2]
  else: var_2 = None
  if param_3 is not None:
    var_3 = list_3[(inputarg // list_3_changes) % len_list_3]
  else: var_3 = None

  return var_1, var_2, var_3

def print_time_taken():
  """
  Print the time taken since the training started
  """

  finishing_time = datetime.now()
  time_taken = finishing_time - starting_time
  d = divmod(time_taken.total_seconds(), 86400)
  h = divmod(d[1], 3600)
  m = divmod(h[1], 60)
  s = m[1]
  print("\nStarted at:", starting_time.strftime(datestr))
  print("Finished at:", datetime.now().strftime(datestr))
  print(f"Time taken was {d[0]:.0f} days {h[0]:.0f} hrs {m[0]:.0f} mins {s:.0f} secs\n")

def inspect_data(data, log_level=1):
  """
  Get the max, mean, and std of data
  """

  max_value = torch.max(-1 * torch.min(data), torch.max(data))
  mean = data.mean()
  std = data.std()
  data = (data - mean) / std
  new_max = torch.max(-1 * torch.min(data), torch.max(data))
  data /= new_max
  if log_level > 0:
    print(f"Normalised evaluations, max_value = {max_value:.3f} (max used = {new_max:.3f}), mean = {mean.item():.3f}, std = {std.item():.3f}, now max value is {torch.max(-1 * torch.min(data), torch.max(data)):.3f}, mean is {data.mean().item():.3f} and std is {data.std().item():.3f}")
  return new_max, mean, std

def print_table(data_dict, timestamp):
  """
  Print a results table
  """

  group_name = timestamp[:8]
  run_name = "run_{0}_A{1}"
  run_starts = run_name.format(timestamp[-5:], "")

  # get the jobs that correspond to this timestamp
  group_path = f"{data_dict['path']}/{data_dict['savepath']}/{group_name}"
  run_folders = [x for x in os.listdir(group_path) if x.startswith(run_starts)]

  job_nums = []

  for folder in run_folders:
    num = folder.split(run_starts)[1] # from run_xx_xx_A5 -> 5"
    job_nums.append(int(num))

  # sort into numeric ascending order
  job_nums.sort()

  # check for failures
  if len(job_nums) == 0:
    print(f"train_nn_evaluator.py warning: print_table found zero trainings matching '{run_starts}'")

  table = []

  for j in job_nums:

    # extract the test report data
    data_dict["savefolder"] = f"{group_name}/{run_name.format(timestamp[-5:], j)}"
    trainer = Trainer(data_dict)
    names, matrix = trainer.load_test_report()
    data_list = list(matrix[-1, :]) # take the last row (most recent data)

    # now assemble this row of the table
    data_row = [timestamp, j]
    if trainer.param_1 is not None: data_row.append(trainer.param_1)
    if trainer.param_2 is not None: data_row.append(trainer.param_2)
    if trainer.param_3 is not None: data_row.append(trainer.param_3)
    for d in data_list: data_row.append(d)
    table.append(data_row)

  # assemble the table column headings
  headings = ["Timestamp     ", "Job"]
  if trainer.param_1 is not None: headings.append(trainer.param_1_name)
  if trainer.param_2 is not None: headings.append(trainer.param_2_name)
  if trainer.param_3 is not None: headings.append(trainer.param_3_name)
  for n in names: headings.append(n)

  # fix later
  program_str = f"Program = {trainer.program}\n\n"

  # now prepare to print the table
  print_str = """""" + program_str
  heading_str = ""
  for x in range(len(headings) - 1): heading_str += "{" + str(x) + "} | "
  heading_str += "{" + str(len(headings) - 1) + "}"
  row_str = heading_str[:]
  heading_formatters = []
  row_formatters = []
  for x in range(len(headings)):
    row_formatters.append("{" + f"{x}:<{len(headings[x]) + 0}" + "}")
    heading_formatters.append("{" + f"{x}:<{len(headings[x]) + 0}" + "}")
  heading_str = heading_str.format(*heading_formatters)
  row_str = row_str.format(*row_formatters)

  # assemble the table text
  print_str += heading_str.format(*headings) + "\n"
  # print(heading_str.format(*headings))
  for i in range(len(table)):
    # check if entry is incomplete
    while len(table[i]) < len(headings): table[i] += ["N/F"]
    for j, elem in enumerate(table[i]):
      if isinstance(elem, float):
        table[i][j] = "{:.3f}".format(elem)
    # print(row_str.format(*table[i]))
    print_str += row_str.format(*table[i]) + "\n"

  # print and save the table
  print("\n" + print_str)
  with open(group_path + f"run_{timestamp[-5:]}_results.txt", 'w') as f:
    f.write(print_str)

class Trainer():

  @dataclass
  class Parameters:

    # training parameters
    num_epochs: int = 5
    
    # key learning hyperparameters
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    loss_style: str = 'MSE'
    batch_size: int = 64
    dropout_prob: float = 0.0

    # specific learning options
    use_combined_loss: bool = False       # combine stockfish and my eval in loss term
    use_sf_eval: bool = True              # use stockfish evaluations to train, rather than my evaluations
    use_sq_eval_sum_loss: bool = False    # if using my eval, include the sum in the loss term (i.e. overall eval)

    # important dataset management options
    checkmate_value: float = 15           # value to clip all evaluations between
    sample_method: str = "default"        # how to sample batches, either default or weighted (for even representation)

    # normalising factors
    norm_mean: float = 0
    norm_std: float = 0
    norm_clip: float = 0

    def update(self, newdict):
      for key, value in newdict.items():
        if hasattr(self, key):
          setattr(self, key, value)
        else: raise RuntimeError(f"incorrect key: {key}")

  def __init__(self, data_dict, log_level=1):
    """
    Initialise the trainer
    """

    self.params = Trainer.Parameters()

    # settings
    self.log_level = log_level
    self.slice_log_rate = 5
    self.test_report_name = "test_report"
    self.test_report_seperator = "---\n"
    self.weighted_sample_num_combined_bins = 20

    # prepare saving and loading, extract info from data_dict
    self.data_dict = data_dict
    loadpath = f"{data_dict['path']}/{data_dict['loadpath']}/{data_dict['loadfolder']}"
    self.dataloader = ModelSaver(loadpath, log_level=data_dict['load_log_level'])
    savepath = f"{data_dict['path']}/{data_dict['savepath']}/{data_dict['savefolder']}"
    self.datasaver = ModelSaver(savepath, log_level=data_dict['save_log_level'])
    self.params.sample_method = data_dict["sample_method"]
    self.characterise_distribution() # extract mean/std dev/distirbution of dataset

    # create variables
    self.net = None
    self.batch_limit = None
    self.program = None
    self.param_1 = None
    self.param_2 = None
    self.param_3 = None
    self.param_1_name = None
    self.param_2_name = None
    self.param_3_name = None
    self.train_loss = []
    self.test_loss = []
    self.test_epochs = []
    self.test_mean_my_diff = []
    self.test_std_my_diff = []
    self.test_mean_sf_diff = []
    self.test_std_sf_diff = []

  def init(self, net, lr=None, weight_decay=None, loss_style=None):
    """
    Initialise the network
    """

    if lr is not None: self.params.learning_rate = lr
    if weight_decay is not None: self.params.weight_decay = weight_decay
    if loss_style is not None: self.params.loss_style = loss_style

    # input critical learning features
    self.net = net
    self.optim = torch.optim.Adam(net.parameters(), lr=self.params.learning_rate, 
                                  weight_decay=self.params.weight_decay)

    if self.params.loss_style.lower() == "mse":
      self.lossfcn = nn.MSELoss()
    elif self.params.loss_style.lower() == "l1":
      self.lossfcn = nn.L1Loss()
    elif self.params.loss_style.lower() == "huber":
      self.lossfcn = nn.HuberLoss()
    else:
      raise RuntimeError(f"train_procedure() error: loss_style = {self.params.loss_style} not recognised")

  def normalise_data(self, data):
    """
    Normalise data based on [max, mean, std]
    """

    clip = self.params.norm_clip
    mean = self.params.norm_mean
    std = self.params.norm_std

    data = torch.clip(data, min=-clip, max=clip)
    data = ((data - mean) / std)

    return data

  def denormalise_data(self, data):
    """
    Undo normalisation based on [max, mean, std]
    """

    clip = self.params.norm_clip
    mean = self.params.norm_mean
    std = self.params.norm_std

    return (data * std) + mean

  def calc_loss(self, net_y, true_y):
    """
    Calculate the loss
    """

    if self.params.use_combined_loss:

      # take the first element in each vector across the whole batch
      loss_board_eval = self.lossfcn(net_y[:, 0], true_y[:, 0])

      # sum the remaining elements
      loss_square_eval = self.lossfcn(net_y[:, 1:], true_y[:, 1:])

      loss = loss_board_eval + loss_square_eval

    elif self.params.use_sf_eval:

      loss = self.lossfcn(torch.sum(net_y, dim=1), true_y)

    else:

      loss = self.lossfcn(net_y.squeeze(1), true_y)

      if self.params.use_sq_eval_sum_loss:

        # constrain also the sum (using MSE error)
        loss += torch.pow(torch.sum(net_y, dim=1) - torch.sum(true_y, dim=1), 2).mean()

    return loss
  
  def characterise_distribution(self):
    """
    Load data on the distribution
    """

    # load the dataset distribution
    dist = self.dataloader.load("distribution", suffix_numbering=False)
    self.bin_counts = dist[0]
    self.bin_ranges = dist[1]
    self.mean_std_max = dist[2]

    if self.params.use_sf_eval:
      self.params.norm_mean = self.mean_std_max[0][0]
      self.params.norm_std = self.mean_std_max[0][1]
      self.params.norm_clip = self.params.checkmate_value
    elif self.params.use_combined_loss:
      self.params.norm_mean = self.mean_std_max[2][0]
      self.params.norm_std = self.mean_std_max[2][1]
      self.params.norm_clip = self.params.checkmate_value
    else:
      raise RuntimeError("Trainer.characterise_distribution() error: currently requires params.use_sf_eval or params.use_combined_loss to be true")

  def create_weighted_sampler(self, evaluations, combine_bins=20):
    """
    Create a weighted sampler based on the current distribution of evaluations
    """

    # hyperparameter, how many bins (/100) to pool on either side of the distribution (recommended: 20)
    combine_bins = self.weighted_sample_num_combined_bins

    if self.params.use_sf_eval:
      raw_bin_counts = self.bin_counts[0]
      raw_bins = self.bin_ranges[0]
    else:
      raise RuntimeError("Trainer.create_weighted_sampler() error: weighted sampling has not been implemented unless self.params.use_sf_eval = True")

    # compute the number of bins we will use for the weighting
    num_bins = len(raw_bin_counts) + 1
    bin_clip = np.max(raw_bins)
    bin_width = (2 * bin_clip) / num_bins

    clip_val = bin_clip - combine_bins * bin_width
    evaluations = torch.clip(evaluations, -clip_val, clip_val)

    # check if we pool at either side of bins, to avoid excessive sampling of rare values
    if combine_bins > 1:
      bin_counts = torch.tensor(np.concatenate([
          [np.sum(raw_bin_counts[:combine_bins])],
          raw_bin_counts[combine_bins:num_bins - combine_bins],
          [np.sum(raw_bin_counts[num_bins - combine_bins:])] 
      ]))
    else:
      bin_counts = torch.tensor(raw_bin_counts)

    # convert to inverse normalised probability
    bin_probs = bin_counts / torch.sum(bin_counts)
    bin_probs = 1.0 / bin_probs
    bin_probs /= torch.sum(bin_probs)

    evaluations += clip_val + 1e-6 # make all positive
    evaluations /= bin_width # scale so bins have integer width
    inds = torch.floor(evaluations).long() # conver bins into indexes -> 1.74 -> bin[1]

    # neither of below should be needed if above is correct
    # inds = torch.clip(inds, 0, len(bin_probs) - 1)
    # inds += 1

    # index each probability bin and assign that prob as the weight
    weight_vec = bin_probs[inds]

    # create the sampler for this batch
    weighted_sampler = torch.utils.data.WeightedRandomSampler(weights=weight_vec, 
                                                              num_samples=self.params.batch_size, 
                                                              replacement=True)
    
    return weighted_sampler

  def prepare_data(self, name, id, to_device=None, return_dataset=False):
    """
    Load and process the data
    """

    # load this segment of the dataset
    dataset = self.dataloader.load(name, id=id)
    data_x = dataset[0]
    if self.params.use_combined_loss:
      data_y = torch.cat((dataset[1].unsqueeze(1), dataset[2]), dim=1)
    elif self.params.use_sf_eval:
      data_y = dataset[1]
    else:
      data_y = dataset[2]

    # clip checkmates
    data_y = torch.clip(data_y, -self.params.checkmate_value, self.params.checkmate_value)

    # normalise y labels
    data_y = self.normalise_data(data_y)

    if to_device is not None:
      data_x = data_x.to(to_device)
      data_y = data_y.to(to_device)

    if return_dataset:
      return data_x, data_y, dataset[1], dataset[2]
    else:
      return data_x, data_y

  def train(self, net=None, epochs=None, norm_factors=None,
            device="cuda", batch_size=None,
            examine_dist=False):
    """
    Perform a training epoch for a given network based on data inputs
    data_x, and correct outputs data_y
    """

    if net is not None: self.init(net)

    if epochs is not None: self.params.num_epochs = epochs

    # save and print the current training hyperparameters
    hyperparam_str = self.save_hyperparameters()
    if self.log_level > 0:
      print(hyperparam_str)

    # move onto the specified device
    self.net.to(device)

    # put the model in training mode
    self.net.train()

    # save the input settings
    self.norm_factors = norm_factors
    if batch_size is not None: self.params.batch_size = batch_size
    
    # each epoch, cover the entire training dataset
    for i in range(self.params.num_epochs):

      print(f"Starting epoch {i + 1} / {self.params.num_epochs}.", flush=True)
      total_batches = 0
      epoch_loss = 0

      # load the dataset in a series of slices (with randomised order)
      train_load_indexes = self.data_dict['train_inds'][:]
      random.shuffle(train_load_indexes)
      for slice_num, j in enumerate(train_load_indexes):

        # load this segment of the dataset
        data_x, data_y, sf_evals, sq_evals = self.prepare_data(self.data_dict['loadname'], j, return_dataset=True)

        if examine_dist:
          import matplotlib.pyplot as plt
          fig, axs = plt.subplots(1, 1)
          axs.hist(torch.flatten(data_y).numpy(), bins=50)
          plt.show()
          inspect_data(data_y)
          continue

        num_batches = len(data_x) // self.params.batch_size
        if num_batches == 0:
          raise RuntimeError("Trainer.train() found num_batches = 0")
        
        # useful for debugging and testing
        if self.batch_limit is not None:
          num_batches = self.batch_limit

        total_batches += num_batches
        rand_idx = torch.randperm(data_x.shape[0])
        avg_loss = 0

        if slice_num % self.slice_log_rate == 0:
          print(f"Epoch {i + 1}. Slice {slice_num + 1} / {len(self.data_dict['train_inds'])} has {num_batches} batches.", end=" ", flush=True)

        if self.params.use_sf_eval:
          weighted_sampler = self.create_weighted_sampler(sf_evals)
        else:
          weighted_sampler = self.create_weighted_sampler(sq_evals)
        
        t1 = time.time()

        # iterate through each batch for this slice of the dataset
        for n in range(num_batches):

          # take a random sample for this batch
          if self.params.sample_method == "default":
            batch_x = data_x[rand_idx[n * self.params.batch_size : (n+1) * self.params.batch_size]]
            batch_y = data_y[rand_idx[n * self.params.batch_size : (n+1) * self.params.batch_size]]
          elif self.params.sample_method == "weighted":
            w_sample_indicies = list(weighted_sampler)
            batch_x = data_x[w_sample_indicies]
            batch_y = data_y[w_sample_indicies]
          else:
            raise RuntimeError(f"Trainer.train() error: sampling method '{self.params.sample_method}' not recognised")
            
          # go to device (small memory footprint, slightly slower)
          batch_x = batch_x.to(device)
          batch_y = batch_y.to(device)

          # use the model for a prediction and calculate loss
          net_y = self.net(batch_x).squeeze(1)
          loss = self.calc_loss(net_y, batch_y)

          # backpropagate
          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

          avg_loss += loss.item()

          # if n % 500 == 0:
          #   print(f"Loss is {(avg_loss / (n + 1)) * 1000:.3f}, epoch {i + 1}, batch {n + 1} / {num_batches}")

        t2 = time.time()

        # this dataset slice is finished
        epoch_loss += avg_loss

        avg_loss = avg_loss / num_batches
        avg_loss = avg_loss * 1e3

        if slice_num % self.slice_log_rate == 0:
          print(f"Loss is {avg_loss:.3f} e-3, took {t2 - t1:.1f}s", flush=True)
    
      # this epoch is finished
      epoch_loss = epoch_loss / total_batches
      epoch_loss = epoch_loss * 1e3
      print(f"Epoch {i + 1} has finished after {total_batches} batches. Overall average loss = {epoch_loss:.3f} e-3.", 
            f"Final slice loss = {avg_loss:.3f}.", flush=True)
      
      # evaluate model performance
      self.train_loss.append(epoch_loss)
      test_report = self.test(epoch=i+1, device=device)

      # print the test report
      print(test_report)

      # save the model after this epoch
      self.datasaver.save(data_dict['savename'], self.net)
      self.datasaver.save(self.test_report_name, txtstr=test_report, txtonly=True)

      # trace and save the traced model as well
      traced_net = torch.jit.trace(self.net, batch_x)
      traced_net.save(self.datasaver.get_current_path() + "/traced_model.pt")

    # finally, return the network that we have trained
    return self.net
  
  def test(self, epoch, device):
    """
    Perform a training epoch for a given network based on data inputs
    data_x, and correct outputs data_y
    """

    # move onto the specified device
    self.net.to(device)

    # put the model in training mode
    self.net.eval()

    total_batches = 0
    test_loss = 0

    test_mean_my_diff = 0
    test_std_my_diff = 0
    test_mean_sf_diff = 0
    test_std_sf_diff = 0

    my_diff_means = []
    my_diff_vars = []
    sf_diff_means = []
    sf_diff_vars = []

    # load the dataset in a series of slices
    for slice_num, j in enumerate(self.data_dict['test_inds']):

      # load this segment of the dataset
      data_x, data_y, dataset_1, dataset_2 = self.prepare_data(self.data_dict['loadname'], j,
                                                               return_dataset=True)

      num_batches = len(data_x) // self.params.batch_size
      if num_batches == 0:
        raise RuntimeError("Trainer.test() found num_batches = 0")
      
      # useful for debugging and testing
      if self.batch_limit is not None:
        num_batches = self.batch_limit

      total_batches += num_batches
      rand_idx = torch.randperm(data_x.shape[0])

      avg_loss = 0

      avg_mean_my_diff = 0
      avg_var_my_diff = 0
      avg_mean_sf_diff = 0
      avg_var_sf_diff = 0

      if slice_num % self.slice_log_rate == 0:
        print(f"Testing. Slice {slice_num + 1} / {len(self.data_dict['test_inds'])} has {num_batches} batches.", end=" ", flush=True)

      if self.params.use_sf_eval:
        weighted_sampler = self.create_weighted_sampler(dataset_1)
      else:
        weighted_sampler = self.create_weighted_sampler(dataset_2)

      t1 = time.time()

      # iterate through each batch for this slice of the dataset
      for n in range(num_batches):

        # take a random sample for this batch
        if self.params.sample_method == "default":
          batch_x = data_x[rand_idx[n * self.params.batch_size : (n+1) * self.params.batch_size]]
          batch_y = data_y[rand_idx[n * self.params.batch_size : (n+1) * self.params.batch_size]]
        elif self.params.sample_method == "weighted":
          w_sample_indicies = list(weighted_sampler)
          batch_x = data_x[w_sample_indicies]
          batch_y = data_y[w_sample_indicies]
        else:
          raise RuntimeError(f"Trainer.train() error: sampling method '{self.params.sample_method}' not recognised")

        # go to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # extract the evaluations from stockfish (sf) and my evaluation function (my)
        sf_evals = dataset_1[rand_idx[n * self.params.batch_size : (n+1) * self.params.batch_size]].to(device)
        my_evals = torch.sum(dataset_2[rand_idx[n * self.params.batch_size : (n+1) * self.params.batch_size]], dim=1).to(device)

        with torch.no_grad():

          # use the model for a prediction and calculate loss
          net_y = self.net(batch_x).squeeze(1)
          loss = self.calc_loss(net_y, batch_y)

        avg_loss += loss.item()

        # convert network prediction to evaluations
        net_y_denorm = self.denormalise_data(net_y)

        # examine performance relative to stockfish evaluation function
        if self.params.use_combined_loss:
          net_sf_eval = net_y_denorm[:, 0]
          net_my_evals = torch.sum(net_y_denorm[:, 1:], dim=1)
          sf_diff = torch.abs(net_sf_eval - sf_evals)
          my_diff = torch.abs(net_my_evals - my_evals)
        else:
          net_y_evals = torch.sum(net_y_denorm, dim=1)
          sf_diff = torch.abs(net_y_evals - sf_evals)
          my_diff = torch.abs(net_y_evals - my_evals)

        avg_mean_sf_diff += torch.mean(sf_diff).item()
        avg_var_sf_diff += torch.var(sf_diff).item()
        avg_mean_my_diff += torch.mean(my_diff).item()
        avg_var_my_diff += torch.var(my_diff).item()

        # save the mean and variance of each batch
        my_diff_means.append(torch.mean(my_diff).item())
        my_diff_vars.append(torch.var(my_diff).item())
        sf_diff_means.append(torch.mean(sf_diff).item())
        sf_diff_vars.append(torch.var(sf_diff).item())

        # if n % 500 == 0:
        #   print(f"Loss is {(avg_loss / (n + 1)) * 1000:.3f}, epoch {i + 1}, batch {n + 1} / {num_batches}")

      t2 = time.time()
      
      # this dataset slice is finished
      test_loss += avg_loss

      test_mean_my_diff += avg_mean_my_diff
      test_std_my_diff += avg_var_my_diff
      test_mean_sf_diff += avg_mean_sf_diff
      test_std_sf_diff += avg_var_sf_diff

      # for printing
      avg_loss = avg_loss / num_batches
      avg_loss = avg_loss * 1e3

      if slice_num % self.slice_log_rate == 0:
        print(f"Loss is {avg_loss:.3f} e-3, took {t2 - t1:.1f}s", flush=True)

    # the test is finished
    test_loss = test_loss / total_batches
    test_loss = test_loss * 1e3

    test_mean_my_diff = test_mean_my_diff / total_batches
    test_std_my_diff = (test_std_my_diff / total_batches) ** 0.5 # convert from variance to std dev
    test_mean_sf_diff = test_mean_sf_diff / total_batches
    test_std_sf_diff = (test_std_sf_diff / total_batches) ** 0.5 # convert from variance to std dev

    # calculate the final mean and variance of each batch
    my_diff_means = np.array(my_diff_means)
    my_diff_overall_mean = np.mean(my_diff_means)
    my_diff_overal_std = np.sqrt(np.mean(my_diff_vars) + np.mean(np.power(my_diff_means - my_diff_overall_mean, 2)))

    sf_diff_means = np.array(sf_diff_means)
    sf_diff_overall_mean = np.mean(sf_diff_means)
    sf_diff_overal_std = np.sqrt(np.mean(sf_diff_vars) + np.mean(np.power(sf_diff_means - sf_diff_overall_mean, 2)))

    self.test_epochs.append(epoch)
    self.test_loss.append(test_loss)
    self.test_mean_my_diff.append(my_diff_overall_mean) #test_mean_my_diff)
    self.test_std_my_diff.append(my_diff_overal_std) #test_std_my_diff)
    self.test_mean_sf_diff.append(sf_diff_overall_mean) #test_mean_sf_diff)
    self.test_std_sf_diff.append(sf_diff_overal_std) #test_std_sf_diff)

    test_report = self.create_test_report()

    print(f"Testing has finished after {total_batches} batches. Overall average loss = {test_loss:.3f} e-3.", 
          f"(mean, std) -> Stockfish({test_mean_sf_diff:.3f}, {test_std_sf_diff:.3f}),",
          f"MyEval({test_mean_my_diff:.3f}, {test_std_my_diff:.3f}) (OLD)",
          flush=True)
    
    print(f"Testing has finished after {total_batches} batches. Overall average loss = {test_loss:.3f} e-3.", 
          f"(mean, std) -> Stockfish({sf_diff_overall_mean:.3f}, {sf_diff_overal_std:.3f}),",
          f"MyEval({my_diff_overall_mean:.3f}, {my_diff_overal_std:.3f})",
          flush=True)
    
    # put the model back into training mode
    self.net.train()
    
    return test_report
  
  def save_hyperparameters(self):
    """
    Save a text file detailing the key hyperparameters used for training
    """

    savestr = """Hyperparemters:\n\n"""

    param_dict = asdict(self.params)
    param_dict.update({
      "dataset_name" : self.data_dict["loadfolder"],
      "save_folder" : self.data_dict["savefolder"],
    })
    
    savestr += str(param_dict).replace(",", "\n") + "\n"

    self.datasaver.save("hyperparemeters", txtstr=savestr, txtonly=True)

    return savestr

  def create_test_report(self):
    """
    Return a string which summarises how the testing is going
    """

    report_str = """"""

    report_str += f"Report for {self.data_dict['savefolder']}\n"
    if self.program is not None: report_str += f"Program = {self.program}\n"

    if self.param_1 is not None:
      report_str += self.test_report_seperator
      if self.param_1 is not None: report_str += f"Param 1 name = {self.param_1_name}, value = {self.param_1}\n"
      if self.param_2 is not None: report_str += f"Param 2 name = {self.param_2_name}, value = {self.param_2}\n"
      if self.param_3 is not None: report_str += f"Param 3 name = {self.param_3_name}, value = {self.param_3}\n"

    if len(self.test_epochs) > 0:
      report_str += self.test_report_seperator

      header_str = f"{'Epoch':<6} | {'Train Loss / e-3':<16} | {'Test Loss / e-3':<16} | {'SF mean':<7} | {'SF std':<7} | {'My mean':<7} | {'My std':<7}"
      row_str = "{0:<6} | {1:<16.3f} | {2:<16.3f} | {3:<7.3f} | {4:<7.3f} | {5:<7.3f} | {6:<7.3f}"

      report_str += header_str + "\n"

      for i, e in enumerate(self.test_epochs):

        report_str += row_str.format(
          e,
          self.train_loss[i],
          self.test_loss[i],
          self.test_mean_sf_diff[i],
          self.test_std_sf_diff[i],
          self.test_mean_my_diff[i],
          self.test_std_my_diff[i]
        ) + "\n"

    return report_str
  
  def load_test_report(self, as_string=False, string_input=None):
    """
    Load data from a test report
    """

    if string_input is None:

      txt = self.datasaver.read_textfile(self.test_report_name)
      if txt is None:
        raise RuntimeError("Trainer.load_test_report() error: no test report found, txt=None")
      if as_string: return txt
      
    else: txt = string_input

    sections = txt.split(self.test_report_seperator)

    found_meta = False
    found_param = False
    found_data = False
    for i, s in enumerate(sections):
      if s.startswith("Report"):
        meta_data = s
        found_meta = True
      elif s.startswith("Param"):
        param_data = s
        found_param = True
      elif s.startswith("Epoch"):
        test_data = s
        found_data = True

    if found_meta:
      lines = meta_data.splitlines()
      for l in lines:
        if l.startswith("Program"):
          self.program = l.split(" = ")[1]

    if found_param:
      """example of string we should have:
      ---
      Param 1 name = learning rate, value = 1e-06
      Param 2 name = use sf eval, value = False
      ---
      """
      lines = param_data.splitlines()
      for i, l in enumerate(lines):
        splits = l.split(", value = ")
        value = splits[1]
        name = splits[0].split(" = ")[1]
        if i == 0:
          self.param_1 = value
          self.param_1_name = name
        elif i == 1:
          self.param_2 = value
          self.param_2_name = name
        elif i == 2:
          self.param_3 = value
          self.param_3_name = name

    if not found_data:
      raise RuntimeError("Trainer.load_test_report() error: 'Epoch' section not found, does test report contain data?")

    # now handle the test data
    lines = test_data.splitlines()

    names = []
    data = []

    datastarted = False

    for i, l in enumerate(lines):

      splits = l.split(" | ")

      if splits[0].startswith("Epoch"):
        for field in splits:
          names.append(field)
        datastarted = True
      elif datastarted:
        new_elem = []
        for field in splits:
          new_elem.append(float(field))
        data.append(new_elem)

    if not datastarted:
      print("Trainer.load_test_report() warning: datastarted=False, no data found in test report")
      return None, None

    # convert into a numpy matrix
    matrix = np.array(data)

    return names, matrix

if __name__ == "__main__":

  # starting time
  starting_time = datetime.now()

  # key default settings
  datestr = "%d-%m-%y_%H-%M" # all date inputs must follow this format

  parser = argparse.ArgumentParser()
  parser.add_argument("-j", "--job", type=int, default=1)                 # job input number
  parser.add_argument("-t", "--timestamp", default=None)                  # timestamp override
  parser.add_argument("-p", "--program", default="default")               # training program name
  parser.add_argument("-lr", "--learning-rate", type=float, default=1e-5) # learning rate during training
  parser.add_argument("-e", "--epochs", type=int, default=5)              # number of training epochs
  parser.add_argument("--device", default="cuda")                         # device to use, "cpu" or "cuda"
  parser.add_argument("--batch-size", type=int, default=64)               # size of learning batches
  parser.add_argument("--loss-style", default="MSE")                      # loss function, 'MSE', 'L1', or 'Huber'
  parser.add_argument("--num-blocks", type=int, default=3)                # number of residual blocks in chess net
  parser.add_argument("--weight-decay", type=float, default=1e-4)         # L2 weight regularisation
  parser.add_argument("--dropout-prob", type=float, default=0.0)          # probability of a node dropping out during training
  parser.add_argument("--torch-threads", type=int, default=1)             # number of threads to allocate to pytorch, 0 to disable
  parser.add_argument("--double-train", action="store_true")              # train on stockfish evaluations afterwards
  parser.add_argument("--use-sf-loss", action="store_true")               # use stockfish evaluation as target
  parser.add_argument("--dataset-name", default=None)                     # name of training dataset to use
  parser.add_argument("--train-num", type=int, default=160)               # number of data files to use for training
  parser.add_argument("--test-num", type=int, default=7)                  # number of data files to use for testing
  parser.add_argument("--batch-limit", type=int, default=0)               # limit batches per file, useful for debugging
  parser.add_argument("--slice-log-rate", type=int, default=5)            # log rate for slices during an epoch
  parser.add_argument("--print-table", action="store_true")               # are we printing a results table

  args = parser.parse_args()

  timestamp = args.timestamp if args.timestamp else datetime.now().strftime(datestr)
  run_name = f"run_{timestamp[-5:]}_A{args.job}"
  group_name = timestamp[:8]

  if args.torch_threads > 0:
    torch.set_num_threads(args.torch_threads)

  # saving/loading information
  data_dict = {
    "path" : "/home/luke/chess",
    "loadpath" : "/python/datasets",
    "loadfolder" : "datasetv8" if args.dataset_name is None else args.dataset_name,
    "loadname" : "data_torch",
    "savepath" : "/python/models",
    "savefolder" : f"{group_name}/{run_name}",
    "savename" : "network",
    "train_inds" : list(range(1, args.train_num + 1)),
    "test_inds" : list(range(args.train_num + 1, 
                             args.train_num + args.test_num + 1)),
    "sample_method" : "weighted",
    "load_log_level" : 0,
    "save_log_level" : 1,
  }

  if args.print_table:
    if args.timestamp is None:
      print("train_nn_evaluator.py error: print_table=True but timestamp not given")
    print_table(data_dict, args.timestamp)
    exit()

  print("Run name:", run_name)
  print("Group name:", group_name)
  print("Program:", args.program)

  # create the trainer object
  trainer = Trainer(data_dict)

  # apply generic settings
  trainer.program = args.program
  trainer.slice_log_rate = args.slice_log_rate
  trainer.params.use_sf_eval = args.use_sf_loss
  trainer.params.use_sq_eval_sum_loss = False
  default_norm_factors = [7, 0, 2.159] # old
  default_norm_factors = [10, 0, 4] # new, for datasetv7

  # apply generic command line settings
  if args.batch_limit > 0:
    trainer.batch_limit = args.batch_limit

  # # create the network based on 19 layer boards
  # newnet = True
  # if newnet:
  #   in_features = 17
  #   out_features = 64
  #   net = FCChessNet(in_features, out_features, num_blocks=args.num_blocks, 
  #                   dropout_prob=args.dropout_prob)
  
  # # or load an existing network
  # else:
  #   group = "01-11-24"
  #   run = "run_10-31_A1"
  #   modelloader = ModelSaver(f"/home/luke/chess/python/models/{group}/{run}")
  #   net = modelloader.load("network", id=None)

  if args.program == "default":

    # create and initialise the network
    in_features = 17
    out_features = 64
    net = FCChessNet(in_features, out_features, num_blocks=args.num_blocks, 
                    dropout_prob=args.dropout_prob)
    trainer.init(
      net=net,
      lr=args.learning_rate,
      weight_decay=args.weight_decay,
      loss_style=args.loss_style,
    )

    # now execute the training
    trainer.train(
      epochs=args.epochs,
      norm_factors=default_norm_factors,
      device=args.device
    )

    print_time_taken()

  elif args.program == "compare_double":

    # define what to vary this training, dependent on job number
    vary_1 = [False, True]
    vary_2 = [1, 3]
    vary_3 = ["l1", "mse"]
    repeats = 1
    trainer.param_1_name = "train my first"
    trainer.param_2_name = "num_blocks"
    trainer.param_3_name = "loss fcn"
    trainer.param_1, trainer.param_2, trainer.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                          param_3=vary_3, repeats=repeats)
    print(trainer.create_test_report())

    # create and initialise the network
    in_features = 17
    out_features = 64
    net = FCChessNet(in_features, out_features, num_blocks=trainer.param_2, 
                    dropout_prob=args.dropout_prob)
    trainer.init(
      net=net,
      lr=args.learning_rate,
      weight_decay=args.weight_decay,
      loss_style=trainer.param_3,
    )

    # are we training on sf evaluations straight away
    trainer.params.use_sf_eval = not trainer.param_1

    print(f"trainer.params.use_sf_eval = {trainer.params.use_sf_eval}")
    print(f"Loss style is {args.loss_style}")
    print(f"Learning rate is {args.learning_rate}")

    # pre-define the number of epochs
    args.epochs = 5

    trainer.train(
      epochs=args.epochs,
      norm_factors=default_norm_factors,
      device=args.device
    )

    # will we double train
    if trainer.param_1:

      print("\n--- Now starting double training ---\n")

      # now train to match the stockfish evaluations
      trainer.params.use_sf_eval = True

      trainer.train(
        epochs=args.epochs,
        norm_factors=default_norm_factors,
        device=args.device
      )

    print_time_taken()

  elif args.program == "compare_lr":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    vary_2 = [False, True]
    vary_3 = None
    repeats = 1
    trainer.param_1_name = "learning rate"
    trainer.param_2_name = "use sf eval"
    trainer.param_3_name = None
    trainer.param_1, trainer.param_2, trainer.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    print(trainer.create_test_report())
    
    # create and initialise the network
    in_features = 17
    out_features = 64
    net = FCChessNet(in_features, out_features, num_blocks=args.num_blocks, 
                    dropout_prob=args.dropout_prob)
    trainer.init(
      net=net,
      lr=trainer.param_1,
      weight_decay=args.weight_decay,
      loss_style=args.loss_style,
    )

    # important! hardcode settings
    args.epochs = 5
    trainer.params.use_sf_eval = trainer.param_2

    # now execute the training
    trainer.train(
      epochs=args.epochs,
      norm_factors=default_norm_factors,
      device=args.device
    )

    print_time_taken()

  elif args.program == "compare_layers":

    # define what to vary this training, dependent on job number
    vary_1 = [1, 2, 3]
    vary_2 = [32, 64, 96]
    vary_3 = None
    repeats = None
    trainer.param_1_name = "num layers"
    trainer.param_2_name = "layer width"
    trainer.param_3_name = None
    trainer.param_1, trainer.param_2, trainer.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    print(trainer.create_test_report())
    
    # create and initialise the network
    in_features = 17
    out_features = 64
    net = FCChessNet(in_features, out_features, num_blocks=trainer.param_1, 
                    dropout_prob=0.0, layer_scaling=trainer.param_2,
                    block_type="simple")
    trainer.init(
      net=net,
      lr=args.learning_rate,
      weight_decay=args.weight_decay,
      loss_style=args.loss_style,
    )

    # important! hardcode settings
    args.epochs = 5
    args.learning_rate = 1e-5
    trainer.params.use_sf_eval = True

    # now execute the training
    trainer.train(
      epochs=args.epochs,
      norm_factors=default_norm_factors,
      device=args.device
    )

    print_time_taken()

  elif args.program == "combined_loss":

    # define what to vary this training, dependent on job number
    vary_1 = [True, False]
    vary_2 = None
    vary_3 = None
    repeats = None
    trainer.param_1_name = "use attention"
    trainer.param_2_name = None
    trainer.param_3_name = None
    trainer.param_1, trainer.param_2, trainer.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    print(trainer.create_test_report())
    
    # create and initialise the networks
    in_features = 17
    if trainer.param_1:
      net = ChessNetAttention(in_features, penultimate_channels=64, num_blocks=1,
                              layer_scaling=64, dropout_prob=0.0)
    else:
      out_features = 65
      net = FCChessNet(in_features, out_features, num_blocks=1, block_type="simple",
                      dropout_prob=0.0, layer_scaling=64,)
    
    # important! hardcode settings
    args.epochs = 5
    args.learning_rate = 1e-5
    trainer.use_combined_loss = True

    trainer.init(
      net=net,
      lr=args.learning_rate,
      weight_decay=args.weight_decay,
      loss_style=args.loss_style,
    )

    # now execute the training
    trainer.train(
      epochs=args.epochs,
      norm_factors=default_norm_factors,
      device=args.device
    )

    print_time_taken()

  elif args.program == "compare_learning":

    # define what to vary this training, dependent on job number
    vary_1 = [1, 2, 3]
    vary_2 = [1e-7, 1e-6, 1e-5]
    vary_3 = None
    repeats = None
    trainer.param_1_name = "num layers"
    trainer.param_2_name = "learning rate"
    trainer.param_3_name = None
    trainer.param_1, trainer.param_2, trainer.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    print(trainer.create_test_report())
    
    # create and initialise the network
    in_features = 17
    out_features = 64
    net = FCChessNet(in_features, out_features, num_blocks=trainer.param_1, 
                    dropout_prob=0.0, layer_scaling=64,
                    block_type="simple")
    
    # apply training settings
    trainer.params.use_sf_eval = True
    trainer.params.num_epochs = 10
    trainer.params.loss_style = "l1"
    trainer.params.learning_rate = trainer.param_2
    trainer.params.sample_method = "weighted"

    # now execute the training
    trainer.train(net=net, device=args.device)

    print_time_taken()

  elif args.program == "example_template":

    # define what to vary this training, dependent on job number
    vary_1 = None
    vary_2 = None
    vary_3 = None
    repeats = None
    trainer.param_1_name = None
    trainer.param_2_name = None
    trainer.param_3_name = None
    trainer.param_1, trainer.param_2, trainer.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    print(trainer.create_test_report())
    
    # create and initialise the network
    in_features = 17
    out_features = 64
    net = FCChessNet(in_features, out_features, num_blocks=args.num_blocks, 
                    dropout_prob=args.dropout_prob)
    trainer.init(
      net=net,
      lr=args.learning_rate,
      weight_decay=args.weight_decay,
      loss_style=args.loss_style,
    )

    # now execute the training
    trainer.train(
      epochs=args.epochs,
      norm_factors=default_norm_factors,
      device=args.device
    )

    print_time_taken()

  else:
    raise RuntimeError(f"train_nn_evaluator.py error: program name of {args.program} not recognised")