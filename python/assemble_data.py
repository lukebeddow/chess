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
import gc
from train_nn_evaluator import EvalDataset

Move = namedtuple("Move",
                  ("move_letters", "eval", "depth", "ranking"))
Position = namedtuple("Position",
                      ("fen_string", "eval", "move_vector"))

# THIS SCRIPT SHOULD ONLY BE RUN IN THE ROOT OF THE chess/ DIRECTORY
# RELATIVE PATHS RELY ON THIS SCRIPT RUNNING FROM THE ROOT

# global variables for saving/loading data
data_path = "python/gamedata/samples"
eval_file_template = "random_n={0}_sample"

def print_move_comparison(my_moves, sf_moves, sortit=True):
  """
  Print a side by side comparison of moves from my engine and from stockfish
  """

  if len(my_moves) != len(sf_moves):
    print(f"The length of the moves was NOT identical, sf_moves = {len(sf_moves)}, my_moves = {len(my_moves)}")

  top_header = "{:^24} | {:^24}"
  bot_header = "{:<5} {:<6} {:<5} {:<4}"
  line_str = "{:<5} {:<6} {:<5} {:<4}"
  sortit = True
  print(top_header.format("Stockfish moves", "My moves"))
  print(bot_header.format("Move", "Eval", "Depth", "Rank"), " | ",
        bot_header.format("Move", "Eval", "Depth", "Rank"))
  
  # loop over and print all of the shared move (sortit=False will FAIL to print properly currently)
  for j in range(len(sf_moves)):
    if sortit:
      found = False
      for k in range(len(my_moves)):
        if my_moves[k].move_letters == sf_moves[j].move_letters:
          found = True
          break
      if not found: continue
    else: k = j
    print(line_str.format(sf_moves[j].move_letters, sf_moves[j].eval, sf_moves[j].depth, sf_moves[j].ranking), " | ",
          line_str.format(my_moves[k].move_letters, my_moves[k].eval, my_moves[k].depth, my_moves[k].ranking))
  
  # print all of the stockfish moves that are not present in my moves
  for j in range(len(sf_moves)):
    found = False
    for k in range(len(my_moves)):
      if my_moves[k].move_letters == sf_moves[j].move_letters:
        found = True
        break
    if not found: 
      print(line_str.format(sf_moves[j].move_letters, sf_moves[j].eval, sf_moves[j].depth, sf_moves[j].ranking), " | ",
              line_str.format("", "", "", ""))
  
  # print all of my moves that are not present in stockfish
  for k in range(len(my_moves)):
    found = False
    for j in range(len(sf_moves)):
      if my_moves[k].move_letters == sf_moves[j].move_letters:
        found = True
        break
    if not found: 
      print(line_str.format("", "", "", ""), " | ",
                line_str.format(my_moves[k].move_letters, my_moves[k].eval, my_moves[k].depth, my_moves[k].ranking))

def get_ranking_difference(my_moves, sf_moves):
  """
  Determine the spearman rank coefficient between the move rankings
  """

  if (len(my_moves) != len(sf_moves)):

    print(f"Spearman ranking failed, lengths are different. my_moves length = {len(my_moves)}, sf_moves length = {len(sf_moves)}")
    print_move_comparison(my_moves, sf_moves, sortit=True)
    
    raise RuntimeError(f"get_ranking_difference() error: length of my_moves ({len(my_moves)}) != length sf_moves ({len(sf_moves)})")
  
    # temporary fix
    if len(my_moves) > len(sf_moves):
      my_moves = my_moves[:len(sf_moves)]
    else:
      sf_moves = sf_moves[:len(my_moves)]

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

  # print(f"The spearman test result is {rank}")

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
  if args.loadfile_index is not None:
    index = int(args.loadfile_index)
  else: index = None
  pydata = modelsaver.load(eval_file_template.format(num_rand), id=index)

  avg_eval_difference = 0
  avg_rank_coefficient = 0
  avg_p_value = 0

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

    # check if stockfish reports a mate
    if sf_eval == "mate":
      if my_eval < -90000 or my_eval > 90000:
        eval_difference = 0
        # print("agreed on a mate")
      else:
        pass
        # print("disagreed on a mate: stockfish says yes, I say no")
    elif my_eval < -90000 or my_eval > 90000:
      # I think its mate, stockfish disagrees
      eval_difference = 30000
      # print("disagreed on a mate: I say yes, stockfish says no")
    else:
      eval_difference = abs(sf_eval - my_eval)

    avg_eval_difference += eval_difference

    # get the spearman rank coefficient
    if len(my_moves) > 2:
      if len(my_moves) != len(sf_moves):
        print("Sample number", i)
        bf.print_FEN_board(position.fen_string)
      rank = get_ranking_difference(my_moves=my_moves, sf_moves=sf_moves)
      rank_co = round(rank.statistic, 3)
      rank_p = round(rank.pvalue, 3)
      # if correlation is negative, invert p-value
      if rank_co < 0:
        rank_p = 1 - rank_p
      avg_rank_coefficient += rank_co
      avg_p_value += rank_p
    else:
      rank_co = "n/a, n<3"
      rank_p = "n/a, n<3"

    # # for debugging
    # if isinstance(rank_co, float) and rank_co < 0:
    #   print_move_comparison(my_moves, sf_moves, sortit=True)

    if args.log_level >= 2:
      print(f"Position {i + 1}/{num_rand}, n={len(sf_moves)}.",
            flush=True, end=" ")
      print(f"eval_difference = {eval_difference}, rank coefficient = {rank_co}, rank pvalue = {rank_p}.",
            flush=True, end=" ")
      print(f"sf best move = {sf_moves[0].move_letters} (eval={sf_moves[0].eval}, depth={sf_moves[0].depth}), my best move = {my_moves[0].move_letters} (eval={my_moves[0].eval}, depth={my_moves[0].depth})",
            flush=True)

  # determine the average difference in evaluations
  avg_eval_difference /= len(pydata)
  avg_rank_coefficient /= len(pydata)
  avg_p_value /= len(pydata)

  if args.log_level >= 1:
    print(f"evaluate_engine() found:")
    print(f"avg_eval_difference = {avg_eval_difference * 1e-3:.3f} (in pawns)")
    print(f"avg_rank_coefficient = {avg_rank_coefficient:.3f} (+1 is perfect agreement, 0 is random)")
    print(f"avg_p_value = {avg_p_value:.3f} (below 0.05 is significant)")

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

  # remove the file to free up memory
  del lines
  gc.collect()

  # prepare a stockfish evaluator
  sf_instance = sf.StockfishWrapper()
  sf_instance.target_depth = 20
  sf_instance.num_lines = 200
  sf_instance.num_threads = args.num_threads
  sf_instance.begin()

  gamedata = []

  # mainly for debugging, best to disable
  save_my_moves_too = False
  test_mine_vs_sf = False

  tstart = time.time()

  # loop over the dataset and create the data structure
  for num, line in enumerate(rand_pos):

    if args.log_level >= 2 or (args.log_level == 1 and num % 10 == 0):
      print(f"Preparing position {num + 1} / {num_rand}", flush=True)

    # extract the fen string CHECK THIS IS ALWAYS SUITABLE
    fen = line.split(" c0 ")[0]

    # get the stockfish evaluations
    sfmoves = sf_instance.generate_moves(fen)

    # check if there are no legal moves in the position
    if len(sfmoves) == 0:
      sf_eval = "mate"
    else:
      sf_eval = sfmoves[0].move_eval

    # save the data in python-only format
    sf_move_data = []
    for i, m in enumerate(sfmoves):
      if i == 0: sf_eval = m.move_eval
      new_move = Move(m.move_letters, m.move_eval, m.depth_evaluated, i)
      sf_move_data.append(new_move)

    # now add to the overall gamedata
    new_position = Position(fen, sf_eval, sf_move_data)

    if args.log_level >= 2:
      print(f"The number of stockfish moves is {len(sfmoves)}")

    # do we process the board with my engine as well (depth=1)
    if save_my_moves_too or test_mine_vs_sf:

      gen_moves_struct = bf.generate_moves_FEN(fen)
      my_moves = gen_moves_struct.get_moves()
      my_eval = gen_moves_struct.get_evaluation() # base board evaluation

      my_move_data = []
      for i, m in enumerate(my_moves):
        new_move = Move(m.to_letters(), m.get_evaluation(), 1, i)
        my_move_data.append(new_move)

      if len(sfmoves) != len(my_move_data):
        print(f"generate_sf_data() warning: length of stockfish moves ({len(sfmoves)}) != length of my moves ({len(my_move_data)})")
      elif args.log_level >= 2:
        print(f"Both stockfish and my method found the same number of moves, n = {len(sfmoves)}")

      if save_my_moves_too:
        new_position_mine = Position(fen, my_eval, my_move_data)
        new_position = [new_position, new_position_mine]
      
      found = np.zeros(len(sfmoves))
      if test_mine_vs_sf:
        for a in range(len(sfmoves)):
          for b in range(len(my_move_data)):
            if sfmoves[a].move_letters == my_move_data[b].move_letters:
              found[a] = 1
        # if every move has not been found in both vectors
        if not found.all():
          print(f"generate_sf_data() warning: Number of moves not found as identical in both stockfish and my method = {len(sfmoves) - np.sum(found):.0f}")
        elif args.log_level >= 2:
          print(f"Stockfish and my method moves were confirmed as identical")

      # # for debugging
      # print(f"My method found {len(my_move_data)} moves:")
      # for i in range(len(my_move_data)):
      #   print(my_move_data[i])
    
    # print(new_position.move_vector)

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

def benchmark_speed(args):
  """
  Measure the speed of generate_moves()
  """

  num_positions = 4096
  num_rand = 4096

  # determine how many positions to load
  files_to_load = ((num_positions - 1) // num_rand) + 1
  
  # load a set of chess positions
  datapath = "/home/luke/chess/python/gamedata/samples"
  eval_file_template = "random_n={0}_sample"
  inds = list(range(1, files_to_load + 1))
  log_level = 1
  dataset = EvalDataset(datapath, eval_file_template.format(num_rand),
                        indexes=inds, log_level=log_level)
  
  # prepare the engine
  engine = tf.Engine()
  
  # loop over each position and measure the time
  t1 = time.process_time()

  for i in range(num_positions):
    gen_moves_struct = bf.generate_moves_FEN(dataset.positions[i].fen_string)

  t2 = time.process_time()

  time_per = (t2 - t1) / num_positions

  print(f"Time per position = {time_per * 1000:.3f} ms, from {num_positions} positions, total time = {(t2 - t1):.1f} s")

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-j", "--job", type=int, default=1)                 # job input number
  parser.add_argument("--evaluate-engine", action="store_true")     # compare my engine against stockfish
  parser.add_argument("--generate-data", action="store_true")       # generate stockfish data
  parser.add_argument("-b", "--benchmark-speed", action="store_true") # measure the speed of 'generate_moves'
  parser.add_argument("--num-rand", type=int, default=10)           # number of random samples
  parser.add_argument("--num-threads", type=int, default=1)         # number of cpu threads to allocate to stockfish
  parser.add_argument("--target-depth", type=int, default=20)       # stockfish target depth for evaluations
  parser.add_argument("--data-file", default="ficsgamesdb_2023_standard2000_nomovetimes.txt") # data file for stockfish generation
  parser.add_argument("--loadfile-index", default=None)             # if loading existing sample files, which index to choose
  parser.add_argument("--use-depth-search", action="store_true")    # use depth search for my engine
  parser.add_argument("--log-level", type=int, default=1)           # how much debug printing to do
  parser.add_argument("--random-seed", type=int, default=None)      # random seed for data generation

  args = parser.parse_args()
  
  if args.generate_data:
    time.sleep(args.job)
    generate_sf_data(args)

  elif args.evaluate_engine:
    evaluate_engine(args)

  elif args.benchmark_speed:
    benchmark_speed(args)

  else:
    print("assemble_data.py error: expected either --generate-data or --evaluate-engine")
    parser.print_usage()