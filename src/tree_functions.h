#ifndef TREE_FUNCTIONS_H_
#define TREE_FUNCTIONS_H_

#include <boost/functional/hash.hpp>
#include <chrono>
#include <thread>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "board_functions.h"

struct Lookup {
    int index;
    bool present;

    int hash_insert() { return index; } // +1
    int key_insert() { return index; }  // +0
    int get_hash() { return index; } // +0
    int get_key() { return index; }  // -1
};

struct NodeEval {
    bool active;
    int max_eval;
};

struct Move {
    int start_sq;
    int dest_sq;
    int move_mod;

    // constructor
    Move() {}
    Move(std::string move_letters);

    // member functions
    int get_start_sq() { return start_sq; }
    int get_dest_sq() { return dest_sq; }
    int get_move_mod() { return move_mod; }
    std::string to_letters();
    void print();
    void set(int start, int dest, int mod) {
        start_sq = start; dest_sq = dest; move_mod = mod;
    }
};

struct TreeKey {
    int layer;
    int entry;
    int move_index;
    int evaluation;

    // member functions
    int get_layer() { return layer; }
    int get_entry() { return entry; }
    int get_move_index() { return move_index; }
    int get_evaluation() { return evaluation; }
    std::vector<int> hash() {
        return std::vector<int> { layer, entry };
    }
    void print();

    // comparison operator for sorting
    bool operator< (const TreeKey& other) const {
        return evaluation < other.evaluation;
    }
};

struct MoveEntry {
    Move move;
    int new_eval;
    std::size_t new_hash;
    int active_move = -1;

    // member functions
    Move get_move() { return move; }
    int get_new_eval() { return new_eval; }
    std::size_t get_new_hash() { return new_hash; }
    void print();
    void print(std::string starter);
    std::string print_move();
    std::string print_eval();

    // comparison operator for sorting
    bool operator< (const MoveEntry& other) const {
        return new_eval < other.new_eval;
    }
};

struct TreeEntry {
    std::vector<Move> parent_moves;
    //TreeKey parent_key;

    std::vector<MoveEntry> move_list;
    std::vector<TreeKey> parent_keys;
    Board board_state;
    std::size_t hash_key;
    int eval;
    bool active;

    int active_move = -2;

    // constructors
    //TreeEntry() {}
    TreeEntry(int size) {
        // constructor can reserve space
        move_list.reserve(size);
    }

    // member functions
    void print();
    void print(std::string starter);
    void print(bool move_list_too);
    void print(bool move_list_too, std::string starter);
    MoveEntry get_move_from_list(int index) { return move_list[index]; }
    std::vector<TreeKey> get_parent_keys() { return parent_keys; }
    Board get_board_state() { return board_state; }
    std::size_t get_hash_key() { return hash_key; }
    int get_eval() { return eval; }
    bool is_active() { return active; }
};

struct TreeLayer {
    std::vector<TreeEntry> board_list;
    std::vector<std::size_t> hash_list;
    std::vector<int> key_list;
    std::vector<TreeKey> finished_games_list;

    std::vector<TreeKey> old_ids;
    std::vector<TreeKey> new_ids;
    std::vector<int> new_ids_groups;
    std::vector<TreeKey> new_ids_parents;
    bool white_to_play;
    int layer_move;

    // constructors
    //TreeLayer() {}
    //TreeLayer(int size, int width_) : board_list(size, TreeEntry(width_)) {
    TreeLayer(int size, int width_, int move, bool layer_wtp) {
        // reserve space in hash and key lists
        board_list.reserve(size);
        hash_list.reserve(size);
        key_list.reserve(size);
        layer_move = move;
        white_to_play = layer_wtp;
    }

    // member functions
    Lookup binary_lookup(std::size_t item, 
        const std::vector<std::size_t>& dictionary);
    int add(TreeEntry entry);
    int find_hash(std::size_t hash_key);
    void print();
    std::size_t get_hash_from_list(int index) { return hash_list[index]; }
    int get_key_from_list(int index) { return key_list[index]; }
    NodeEval find_max_eval(int entry, int sign, int active_move, int depth);
    void add_finished_games(std::vector<TreeKey>& id_list);
    std::vector<TreeKey> remove_duplicates(std::vector<TreeKey>& old_list);

    boost::hash<std::vector<int>> treeKeyHash;
};

// testing threading
struct ThreadOut {
    TreeKey id;
    Board board;
    generated_moves_struct gen_moves;
};

class LayeredTree
{
    /* This class represents a move tree in a chess game */
public:
    /* Member functions */
    LayeredTree();
    //~LayeredTree();
    LayeredTree(int width);
    LayeredTree(Board board, bool white_to_play, int width);
    void init(int width);
    int set_root();
    int set_root(Board board, bool white_to_play);
    int get_boards_checked() { return boards_checked_; }
    void add_layer(int size, int layer_move, bool layer_wtp);
    void remove_layer();
    void set_width(int width) { width_ = width; }
    std::shared_ptr<TreeLayer> get_layer_pointer(int layer);
    TreeKey add_move(TreeKey parent_key, move_struct& move);
    void add_board_replies(const TreeKey& parent_key, 
        generated_moves_struct& gen_moves);
    bool grow_tree();
    void print();
    void print(int layers);
    void print_old_ids();
    void print_new_ids();
    void print_new_ids(int max);
    bool check_game_continues();
    void advance_ids();
    void step_forward();
    void cascade();
    bool next_layer();
    bool next_layer(int layer_width, int num_cpus);
    void limit_prune();
    void recursive_prune(std::vector<TreeKey>& id_set, TreeKey parent,
        int k, int kmax, bool deactivate);
    std::vector<TreeKey> cutoff_prune();
    std::vector<TreeKey> cutoff_prune(bool deactivate);
    void target_prune(int target);
    std::vector<MoveEntry> get_best_moves();
    std::vector<MoveEntry> get_best_moves(TreeKey node);
    std::vector<MoveEntry> get_dead_moves(TreeKey node);
    void print_best_move();
    void print_best_moves();
    void print_best_moves(TreeKey node);
    void print_best_moves(bool dead_nodes);
    void print_best_moves(TreeKey node, bool dead_nodes);
    void print_boards_checked();
    void print_node_board(TreeKey node);
    std::vector<TreeKey> my_sort(std::vector<TreeKey>& vec);
    std::vector<TreeKey> remove_duplicates(std::vector<TreeKey>& vec);
    void deactivate_node(TreeKey& node);
    MoveEntry search(int depth);

    bool is_active(MoveEntry& move) {
        return move.active_move == active_move_;
    }

    // for testing
    void print_id_list(std::vector<TreeKey> id_list);
    bool test_dictionary();
    TreeKey find_hash(std::size_t hash, int layer);
    TreeKey search_hash(std::size_t hash);

    void mark_for_deactivation(std::vector<TreeKey> vec, bool deactivate);
    void deactivate_nodes();

    // testing threading
    //void generate_thread(std::vector<ThreadOut>& output, bool white_to_play);
    bool grow_tree_threaded(int num_cpus);

    /* Class variables */
    int current_move_;                          // which move are we on at layer 0
    bool fresh_tree_;
    //std::hash<int[120]> hash_func_;             // board hash function
    boost::hash<int_P[120]> hash_func_;           // board hash function
    int width_;
    int default_cpus_;
    std::vector<TreeKey> old_ids_;
    std::vector<TreeKey> new_ids_;

    struct {
        int base_allowance;
        int decay;
        int min_allowance;
        int minimum_group_size;
        int max_iterations;
        bool use_minimum_group_size;
        float wiggle;
    } prune_params_;

    std::vector<int> new_ids_groups_;
    std::vector<TreeKey> new_ids_parents_;

    bool old_ids_wtp_;                          // old ids white to play
    bool new_ids_wtp_;
    std::vector<TreeKey> finishedGames_;
    int boards_checked_ = 0;
    TreeKey root_;
    bool root_wtp_;
    int root_outcome_;
    int active_move_;
    std::vector<std::shared_ptr<TreeLayer>> layer_pointers_;

    // testing new deactivation
    std::vector<std::size_t> deactivation_hash;
    std::vector<TreeKey> deactivation_nodes;
    std::vector<std::size_t> reactivation_hash;


};

class Engine
{
    /* chess engine class that you can play against */
public:

    std::unique_ptr<LayeredTree> tree_pointer;

    // settings struct
    struct Settings {
        int start_width;
        int end_width;
        int width_decay;

        std::vector<int> width_vector;
        std::vector<int> prune_vector;
        int target_boards;
        int target_time;

        int width;
        int depth;
        int first_layer_width_multiplier;
        int prune_target;
        int num_cpus;
    } settings;

    // details struct
    struct Details {
        int boards_checked;
        double ms_per_board;
        long total_ms;
    } details;

    std::chrono::time_point<std::chrono::steady_clock> start_, end_;

    float board_ms_;
    int print_level; //0=none, 1=minimum, 2=roundup, 3=layer_by_layer, 4=all

    Engine();
    void set_width(int width);
    void set_depth(int depth);
    void set_prune_target(int prune_target);
    void set_first_layer_width_multiplier(int flwm);
    void print_startup(std::unique_ptr<LayeredTree>& tree_ptr);
    void print_layer(std::unique_ptr<LayeredTree>& tree_ptr, int layer);
    void print_roundup(std::unique_ptr<LayeredTree>& tree_ptr);
    void print_responses(std::unique_ptr<LayeredTree>& tree_ptr);
    void calibrate(Board board, bool white_to_play);
    void calculate_settings(double target_time);
    Move generate(Board board, bool white_to_play, double target_time);
};

class GameBoard
{
    /* Chess game board */
public:

    Board board;
    bool white_to_play;
    std::vector<std::string> move_list;
    int outcome; // 0=play continues, 1=checkmate, 2=draw
    std::unique_ptr<Engine> engine_pointer;
    bool print;

    GameBoard();
    GameBoard(Board board, bool white_to_play);
    GameBoard(std::vector<std::string> move_list);
    void init();
    void init(Board myboard, bool mywhite_to_play);
    void init(std::vector<std::string> move_list);

    // get member variables
    Board get_board() { return board; }
    bool get_white_to_play() { return white_to_play; }
    std::vector<std::string> get_move_list() { return move_list; }
    std::string get_last_move() { 
        if (move_list.size() == 0) return "none";
        return move_list[move_list.size() - 1]; 
    }
    std::string get_outcome() {
        if (outcome == 0) return "continue";
        else if (outcome == 1)
            return white_to_play ? "black wins" : "white wins";
        else return "draw";
    }
    bool check_promotion(std::string move);
    int get_square_colour(std::string square);
    int get_square_raw_value(int square_num);
    int get_square_raw_value(std::string square);
    std::string get_square_piece(int square_num);
    std::string get_square_piece(std::string square);

    // helpful functions
    bool move(std::string move);
    void undo(int moves_undone);
    void reset();
    void reset(std::vector<std::string> mymove_list);
    void reset(Board board, bool white_to_play);

    // play against the computer
    std::string get_engine_move();
    std::string get_engine_move(int target_time);
    std::string get_engine_move_no_GIL(int target_time);
    
};

class Game
{
    /* play a game of chess against the computer */
public:
    std::unique_ptr<Engine> engine_pointer;
    std::vector<Move> moves;
    std::vector<std::string> move_letters;
    Board board;
    bool quit = false;

    Game();
    void play_terminal(bool human_first);
    Move get_human_move(Board board, bool white_to_play);

    bool is_move_legal(Board board, bool white_to_play, std::string move_string);
    Board human_move(Board board, bool white_to_play, std::string move_string);
    Board engine_move(Board board, bool white_to_play);
    std::string get_last_move();

};

// class GilManager
// {
// public:
//     GilManager()
//     {
//         mThreadState = PyEval_SaveThread();
//     }

//     ~GilManager()
//     {
//         if (mThreadState)
//             PyEval_RestoreThread(mThreadState);
//     }

//     GilManager(const GilManager&) = delete;
//     GilManager& operator=(const GilManager&) = delete;
// private:
//     PyThreadState* mThreadState;
// };

//int start()
//{
//    GilManager g;
//    // Enable run-time memory leak check for debug builds.
//#if defined(DEBUG) | defined(_DEBUG)
//    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
//#endif
//    WinMain(GetModuleHandle(0), 0, 0, SW_SHOW);
//
//    return 0;
//}

#endif