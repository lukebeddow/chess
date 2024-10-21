import modules.board_module as bf
import modules.tree_module as tf
import modules.stockfish_module as sf
from ModelSaver import ModelSaver
import random
from dataclasses import dataclass
from collections import namedtuple
import time
import argparse
import numpy as np
from scipy.stats import spearmanr

Move = namedtuple("Move",
                  ("move_letters", "eval", "depth", "ranking"))
Position = namedtuple("Position",
                      ("fen_string", "eval", "move_vector"))

# THIS SCRIPT SHOULD ONLY BE RUN IN THE ROOT OF THE chess/ DIRECTORY
# RELATIVE PATHS RELY ON THIS SCRIPT RUNNING FROM THE ROOT

# global variables for saving/loading data
data_path = "python/gamedata/samples"
eval_file_template = "random_n={0}_sample"

def get_ranking_difference(my_moves, sf_moves):
  """
  Determine the spearman rank coefficient between the move rankings
  """

  if (len(my_moves) != len(sf_moves)):
    print(f"Spearman ranking failed, lengths are different. my_moves length = {len(my_moves)}, sf_moves length = {len(sf_moves)}")
    if len(my_moves) > len(sf_moves):
      sf_moves = sf_moves[:len(my_moves)]
    else:
      my_moves = my_moves[:len(sf_moves)]
  
  # get the moves into relative orders, taking stockfish as ground truth
  sf_rank = list(range(len(sf_moves)))

  my_rank = []
  for move in my_moves:
    for i in range(len(sf_moves)):
      if move.move_letters.lower() == sf_moves[i].move_letters.lower():
        my_rank.append(i)
  
  # convert into numpy arrays
  sf_np = np.array(sf_rank)
  my_np = np.array(my_rank)

  rank = spearmanr(sf_np, my_np)

  print(f"The spearman rank score is {rank}")

  return rank

def evaluate_engine(args):
  """
  Evaluates my engine by comparing it against stockfish. The currently compiled
  version of my engine is used. The stockfish data should be given.
  """

  modelsaver = ModelSaver(data_path, log_level=args.log_level)

  use_depth_search = args.use_depth_search
  if use_depth_search:
    engine = tf.Engine()
  if args.log_level >= 2:
    print("Use depth search =", use_depth_search)

  # load the stockfish data
  num_rand = args.num_rand
  pydata = modelsaver.load(eval_file_template.format(num_rand))

  avg_eval_difference = 0

  # loop over the positions and evaluate
  for i, position in enumerate(pydata):

    # extract key data
    fen = position.fen_string
    sf_eval = position.eval
    sf_moves = position.move_vector

    # get my move evaluations (in nice namedtuple format)
    my_moves = []
    if use_depth_search:
      my_moves_raw = engine.generate_engine_moves_FEN(fen)
      my_eval = my_moves_raw[0].get_new_eval()
      for j, move in enumerate(my_moves_raw):
        new_move = Move(move.to_letters(), move.get_new_eval(), 
                        move.get_depth_evaluated(), j)
        my_moves.append(new_move)
    else:
      gen_moves_struct = bf.generate_moves_FEN(fen)
      my_moves_raw = gen_moves_struct.get_moves()
      my_eval = gen_moves_struct.get_evaluation() # before move generation
      for j, move in enumerate(my_moves_raw):
        new_move = Move(move.to_letters(), move.get_evaluation(), 1, j)
        my_moves.append(new_move)

    eval_difference = abs(sf_eval - my_eval)
    avg_eval_difference += eval_difference

    print(my_moves)
    print(sf_moves)

    rank = get_ranking_difference(my_moves=my_moves, sf_moves=sf_moves)

    if args.log_level >= 2:
      print(f"Position {i + 1}/{num_rand}, eval_difference = {eval_difference}, sf best move = {sf_moves[0].move_letters} (eval={sf_moves[0].eval}, depth={sf_moves[0].depth}), my best move = {my_moves[0].move_letters} (eval={my_moves[0].eval}, depth={my_moves[0].depth})",
            flush=True)

  # determine the average difference in evaluations
  avg_eval_difference /= len(pydata)

  if args.log_level >= 1:
    print(f"evaluate_engine() found avg_eval_difference = {avg_eval_difference * 1e-3:.3f} (in pawns)")

  return avg_eval_difference

def generate_sf_data(args):
  """
  Generate stockfish evaluations for random positions
  """

  # filepath to load
  filepath = "./data/"
  filename = "ficsgamesdb_2023_standard2000_nomovetimes_400127.txt"
  filename = args.data_file

  if args.log_level >= 1:
    print(f"Preparing to generate data from file: {filepath + filename}")

  # read in the text data
  with open(filepath + filename, "r") as f:
    lines = f.readlines()

  # remove any lines which are empty
  trimmed_lines = []
  for l in lines:
    if l != "\n": trimmed_lines.append(l)
  lines = trimmed_lines

  # now randomly select games
  num_rand = args.num_rand
  rand_seed = args.random_seed
  num_pos = len(lines)
  if args.log_level >= 2:
    print(f"The number of possible positions to sample from is {num_pos}")
  if num_pos < num_rand:
    raise RuntimeError(f"generate_sf_data() error: num_pos ({num_pos}) is less than num_rand ({num_rand})")
  random.seed(rand_seed)
  rand_pos = random.sample(lines, num_rand)

  # prepare a stockfish evaluator
  sf_instance = sf.StockfishWrapper()
  sf_instance.target_depth = 20
  sf_instance.num_lines = 200
  sf_instance.num_threads = args.num_threads
  sf_instance.begin()

  gamedata = []
  sf_only = True

  tstart = time.time()

  # loop over the dataset and create the data structure
  for num, line in enumerate(rand_pos):

    if args.log_level >= 2 or (args.log_level == 1 and num % 10 == 1):
      print(f"Preparing position {num + 1} / {num_rand}", flush=True)

    # extract the fen string CHECK THIS IS ALWAYS SUITABLE
    fen = line.split(" c0 ")[0]

    # get the stockfish evaluations
    sfmoves = sf_instance.generate_moves(fen)
    sf_eval = sfmoves[0].move_eval

    # if not sf_only:
    #   # get my evaluation
    #   gen_moves_struct = bf.generate_moves_FEN(fen)
    #   my_moves = gen_moves_struct.moves
    #   my_eval = gen_moves_struct.base_evaluation # before move generation

    # save the data in python-only format
    sf_move_data = []
    print(f"The number of sfmoves is {len(sfmoves)}")
    for i, m in enumerate(sfmoves):
      if i == 0: sf_eval = m.move_eval
      new_move = Move(m.move_letters, m.move_eval, m.depth_evaluated, i)
      sf_move_data.append(new_move)

    # if not sf_only:
    #   my_move_data = []
    #   for i, m in enumerate(my_moves):
    #     new_move = Move(m.to_letters(), m.evaluation, 1, i)
    #     my_move_data.append(new_move)

    # now add to the overall gamedata
    new_position = Position(fen, sf_eval, sf_move_data)

    # print(new_position.move_vector)
    
    # if not sf_only:
    #   new_position_mine = Position(fen, my_eval, my_move_data)
    #   new_position = [new_position, new_position_mine]

    gamedata.append(new_position)

  # save the random sample in a text file
  rand_pos_str = ""
  for r in rand_pos:
    rand_pos_str += r
  modelsaver = ModelSaver(data_path, log_level=args.log_level)
  modelsaver.save(eval_file_template.format(num_rand), gamedata, txtstr=rand_pos_str)

  tend = time.time()

  if args.log_level >= 1:
    print(f"Finished assembling data, total time taken = {tend - tstart:.1f} seconds ({(tend - tstart) / num_rand:.1f}s per position)")

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--evaluate-engine", action="store_true")     # compare my engine against stockfish
  parser.add_argument("--generate-data", action="store_true")       # generate stockfish data
  parser.add_argument("--num-rand", type=int, default=10)           # number of random samples
  parser.add_argument("--num-threads", type=int, default=1)         # number of cpu threads to allocate to stockfish
  parser.add_argument("--target-depth", type=int, default=20)       # stockfish target depth for evaluations
  parser.add_argument("--data-file", default="ficsgamesdb_2023_standard2000_nomovetimes.txt") # data file for stockfish generation
  parser.add_argument("--use-depth-search", action="store_true")    # use depth search for my engine
  parser.add_argument("--log-level", type=int, default=2)           # how much debug printing to do
  parser.add_argument("--random-seed", type=int, default=None)      # random seed for data generation

  args = parser.parse_args()

  if args.evaluate_engine:
    evaluate_engine(args)
  
  if args.generate_data:
    generate_sf_data(args)