#ifndef STOCKFISH_H_
#define STOCKFISH_H_

#include <iostream>
#include <chrono>
#include <boost/process.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

namespace bp = boost::process;

struct StockfishMove {
  std::string move_letters;   // move string, eg e2e4
  int move_eval;              // evaluation, in 1000th of a pawn
  int depth_evaluated;        // depth of evaluation
  int move_placement;         // 1=best move, 2=second best, etc

  void print() {
    std::cout << "SF-Move: " << move_letters << ", with eval = "
      << float(move_eval) * 1e-3 << " (" << move_eval
      << "), depth_evaluated = " << depth_evaluated
      << ", move_placement = " << move_placement << "\n";
  }
};

struct StockfishWrapper {

  bp::ipstream is;
  bp::opstream os;
  std::unique_ptr<bp::child> c;

  // default settings
  int target_depth = 20;  // ply depth to search to, overridden by exact_time
  int exact_time = -1;    // exact time in ms to search, -1 disables
  int num_threads = 1;    // number of cpu threads
  int num_lines = 200;      // number of best move lines to output
  int elo_value = 0;      // set 0 to disable limiting, min=1320, max=3190
  int hash_size = 100;    // size of hash table in MB

  // functions
  void begin();
  void send_command(const std::string& command);
  std::vector<StockfishMove> read_best_at_depth(int depth = 0);
  std::vector<StockfishMove> generate_moves(std::string fen);

};

#endif  // #ifndef STOCKFISH_H_