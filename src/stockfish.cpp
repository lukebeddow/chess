#include "stockfish.h"

void StockfishWrapper::init()
{
  /* initialise the stockfish engine in its own thread as a child process */

  c = std::make_unique<bp::child>("./stockfish/stockfish", bp::std_in < os, 
           bp::std_out > is);

  os << "uci" << std::endl;

  // initial board position as an fen string
  std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  // input settings
  os << "setoption name Threads value " << num_threads << std::endl;
  os << "setoption name MultiPV value " << num_lines << std::endl;

  // set a specific elo (min: 1320, max:3190)
  if (elo_value != 0) {
    if (elo_value < 1320 or elo_value > 3190) {
      throw std::runtime_error("stockfish.cpp elo value outside 1320-3190 range");
    }
    os << "setoption name UCI_Elo value " << elo_value << std::endl;
    os << "setoption name UCI_LimitStrength value true" << std::endl;
  }

  os << "isready" << std::endl;

  // exhaust the output
  std::string line;
  std::string move_string;
  while (getline(is, line)) {
    if (!line.compare(0, 5, "uciok")) {
      move_string = line;
      break;
    }
  }
}

void StockfishWrapper::send_command(const std::string& command) 
{
  /* send a command into the child process */

  if (c and c->running()) {
    os << command << std::endl;
    os.flush();  // ensure the command is sent immediately
  } 
  else {
    std::cerr << "Child process is not running!" << std::endl;
  }
}

std::vector<StockfishMove> StockfishWrapper::read_best_at_depth(int depth) 
{
  /* read lines of output from the engine */
  
  std::string line;
  std::vector<StockfishMove> moves;

  while (std::getline(is, line)) {

    char c;
    int word_start = 0;
    int word_end = 0;
    std::vector<std::string> line_words;

    // // debugging
    // std::cout << "True line: " << line << "\n";

    for (int i = 0; i < line.size(); i++) {
      if (line[i] == ' ') {
        word_end = i;
        line_words.push_back(line.substr(word_start, word_end - word_start));
        word_start = i + 1;
      }
      else if (i == line.size() - 1) {
        line_words.push_back(line.substr(word_start, word_end - word_start));
      }
    }

    // // debugging: print out information
    // std::cout << "Line: ";
    // for (int i = 0; i < line_words.size(); i++) {
    //   std::cout << line_words[i] << " ";
    // }
    // std::cout << std::endl;

    if (line_words.size() > 21) {

      if (line_words[0] == "info" and line_words[1] == "depth") {

        int new_depth = std::stoi(line_words[2]);
        if (new_depth > depth) {
          depth = new_depth;
        }

        if (new_depth == depth) {

          // add a move
          StockfishMove m;
          m.move_letters = line_words[21];
          m.move_eval = std::stoi(line_words[9]);
          m.depth_evaluated = depth;
          m.move_placement = moves.size() + 1;
          moves.push_back(m);

        }
        else if (new_depth > depth) {
          break;
        }
      }
    }
    else if (line_words.size() > 1) {
      if (line_words[0] == "bestmove") {
        break;
      }
      else if (line_words[0] == "uciok") {
        break;
      }
    }
    // else {
    //   break;
    // }
  }

  // std::cout << "Finished finding best moves at depth " << depth << "\n";

  return moves;
}

std::vector<StockfishMove> StockfishWrapper::generate_moves(std::string fen)
{
  /* generate moves for a given position using stockfish */

  bool printout = false;

  std::chrono::time_point<std::chrono::steady_clock> start_, end_;
  start_ = std::chrono::steady_clock::now();

  // input the given position
  os << "position fen " << fen << std::endl;

  // determine whether to evaluate with depth or time
  if (exact_time > 1) {
    os << "go movetime " << exact_time << std::endl;
  }
  else {
    os << "go depth " << target_depth << std::endl;
  }

  // get the best moves out of the engine
  std::vector<StockfishMove> moves;
  if (exact_time > 0) {
    throw std::runtime_error("giving stockfish exact time not yet added to generate_moves()");
    // int move_length = 0;
    // bool started = false;
    // while (true) {
    //   std::vector<StockfishMove> new_moves = read_best_at_depth();
    //   if (new_moves.size() > 0) {
    //     started = true;
    //     move_length = new_moves.size();
    //     moves.clear();
    //     for (int i = 0; i < new_moves.size(); i++) {
    //       moves[]
    //     }
    //   }
    // }
  }
  else {
    if (printout) {
      std::cout << "About to read best moves at depth " << target_depth << "\n";
    }
    moves = read_best_at_depth(target_depth);
  }

  if (printout) {
    std::cout << "Stockfish has found the following "
      << moves.size() << " best moves:\n";
    for (int i = 0; i < moves.size(); i++) {
      moves[i].print();
    }
  }

  // end the clock
  end_ = std::chrono::steady_clock::now();

  // determine the timings overall and per board
  long total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
  if (printout) {
    std::cout << "Total time taken = " << total_ms << " ms\n";
  }

  return moves;

  std::string line;
  std::string move_string;
  
  while (getline(is, line)) {
    std::cout << line << std::endl;
  }

  // while (getline(is, line)) {
  //     if (!line.compare(0, 8, "bestmove")) {
  //         move_string = line;
  //         break;
  //     }
  // }
  // // Delete the "bestmove" part of the string and get rid of any trailing characters divided by space
  // move_string = move_string.substr(9, move_string.size()-9);
  // vector<string> mv;
  // boost::split(mv, move_string, boost::is_any_of(" "));
  // cout << "Stockfish move: " << mv.at(0) << endl;

  std::cout << move_string << "\n";
}

int main() {

  StockfishWrapper sf;

  sf.init();

  std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  sf.generate_moves(fen);

  return 0;
}