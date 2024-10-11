#ifndef BOARD_FUNCTIONS_H_
#define BOARD_FUNCTIONS_H_

#include <iostream>
#include <vector>
//#include <Windows.h>

typedef int int_P; // piece int, signed, -7 to +7 (int8_t)
typedef int int_B; // board array int, unsigned, 0 to 120

struct Board {
    int_P arr[120];
    int_P look(int_B i) { return arr[i]; }
    void set(int_B i, int_P val) { arr[i] = val; }
};

struct piece_moves_struct {
    /* Data structure for use with the how_piece_moves function */
    std::vector<int> directions;
    int depth = 0;
};

struct PIECE_MOVES {
    int pawn[3] = { -9, -10, -11 };
    int knight[8] = { -21, -19, -12, -8, 8, 12, 19, 21 };
    int bishop[4] = { -11, -9, 9, 11 };
    int rook[4] = { -10, -1, 1, 10 };
    int queen[8] = { -11, -10, -9, -1, 1, 9, 10, 11 };
    int king[8] = { -11, -10, -9, -1, 1, 9, 10, 11 };
};

struct piece_attack_defend_struct {
    /* Data structure to store the lists generated by the function
       piece_attack_defend */

       // four key lists for piece analysis
    std::vector<int> attack_list;
    std::vector<int> attack_me;
    std::vector<int> defend_me;
    std::vector<int> piece_view;

    // piece evaluation
    int evaluation = 0;

    // member functions
    std::vector<int> get_attack_list() { return attack_list; };
    std::vector<int> get_attack_me() { return attack_me; };
    std::vector<int> get_defend_me() { return defend_me; };
    std::vector<int> get_piece_view() { return piece_view; };
    int get_evaluation() { return evaluation; };

};

struct total_legal_moves_struct {
    /* Data structure to store the lists generated by the function
    total_legal_moves. This is one piece_attack_defend_struct for
    every piece on the board */

    // create a 64 element vector of pad structs, uninitialised
    std::vector<piece_attack_defend_struct> square_info =
        std::vector<piece_attack_defend_struct>(64);

    // other variables
    int evaluation = 0;
    int outcome = 0;
    int phase;
    int phase_adjust;
    std::vector<int> legal_moves;

    // member functions
    piece_attack_defend_struct get_square(int i) { return square_info[i]; };
    piece_attack_defend_struct get_from_index(int ind) { return square_info[(ind - 21) - 2 * ((ind / 10) - 2)]; };
    void set_from_index(int ind, piece_attack_defend_struct pad_struct) {
        square_info[(ind - 21) - 2 * ((ind / 10) - 2)] = pad_struct;
        return;
    };
    std::vector<int> get_legal_moves() { return legal_moves; };
    float get_evaluation() { return evaluation; };
    int get_outcome() { return outcome; };
    int get_phase_adjust() { return phase_adjust; }
    int get_phase() { return phase; }
    int get_old_value(int view_ind) { return square_info[view_ind].evaluation; }
    std::vector<int> get_piece_view(int view_ind) { 
        return square_info[view_ind].piece_view;
    }
};

struct phase_struct {
    /* Data structure to store information about the phase of the
    game, and what modifiers to use for eval_piece */

    int phase;                          // game phase, 1=early game, 2=mid, 3=late
    int evaluation_adjust;              // change to evaluation based on phase

    // bonus modifiers the pieces can get if they attack a lot of squares
    int pawn_mod;
    int knight_mod;
    int bishop_mod;
    int rook_mod;
    int queen_mod;
    int king_mod;

    int king_vunerability_mod;          // approx max penalty to the king if left out in centre of board
    int piece_defended_boost;           // boost applied if a piece is defended
    int pawn_pawn_defender_boost;       // additional boost if a pawn defends a pawn (applied twice for protected past pawn)
    int passed_pawn_bonus;              // bonus for past pawns (fully given if one square away, then 6/7ths for 2 sqs etc...)
    int tempo_bonus;                    // bonus applied for each point of tempo gained
    int castle_bonus;                   // bonus score gained for castling

    // null evaluation modifiers
    int null_attack;  // divisor to reduce null evaluations when we are attacking
    int null_defend;  // divisor to reduce null evaulations when we are defending
    int null_active;  // active modifier that is actually used

    int get_phase() { return phase; }
    int get_eval_adjust() { return evaluation_adjust; }

};

struct verified_move {
    bool legal;
    int start_sq;
    int dest_sq;
    int move_mod;
};

struct move_struct {
    /* This struct represents a move that has been made on a board */

    int evaluation;     // board evaluation after this move
    std::size_t hash;   // hash key for this board
    int start_sq;       // starting square of the move
    int dest_sq;        // destination square of the move
    int move_mod;       // move modifier of the move
    Board board;        // resultant board

    // member functions
    int get_evaluation() { return evaluation; }
    std::size_t get_hash() { return evaluation; }
    int get_start_sq() { return start_sq; }
    int get_dest_sq() { return dest_sq; }
    int get_move_mod() { return move_mod; }
    Board get_board() { return board; }
};

struct generated_moves_struct {
    /* This struct stores a list of moves generated in a position */

    bool game_continues;        // is the starting board a finished game
    bool mating_move = false;   // is there a mating move in the position
    int base_evaluation;        // evaluation of the starting board
    int outcome = 0;            // game outcome (0=none, 1=checkmate, 2=draw)
    Board base_board;
    bool white_to_play;
    std::vector<move_struct> moves;

    // member functions
    bool does_game_continue() { return game_continues; }
    int get_evaluation() { return base_evaluation; }
    int get_outcome() { return outcome; }
    Board get_board() { return base_board; }
    std::vector<move_struct> get_moves() { return moves; }
    int get_length() { return moves.size(); }
    bool is_mating_move() { return mating_move; }
};

/* Globals */

constexpr int WHITE_MATED = -100100;
constexpr int BLACK_MATED = 100100;

//struct {
//    //              A  B  C  D  E  F  G  H
//    int[64] pawn ={ 1, 2, 3, 4, 5, 6, 7, 8,    // 1
//                    0, 0, 0, 0, 0, 0, 0, 0,    // 2
//                    0, 0, 0, 0, 0, 0, 0, 0,    // 3
//                    1, 1, 2, 2, 2, 2, 1, 1,    // 4
//                    1, 1, 2, 2, 2, 2, 1, 1,    // 5
//                    1, 1, 2, 2, 2, 2, 1, 1,    // 6
//                    1, 1, 2, 2, 2, 2, 1, 1,    // 7
//                    1, 1, 2, 2, 2, 2, 1, 1 };  // 8
//    
//};

/* Functions */

#if defined(LUKE_PYBIND)
    void py_print_board(Board& board, bool neat);
    void py_print_board(Board& board);
#endif

Board create_board();
Board create_board(std::vector<std::string> moves);
bool check_board(Board& board);
void print_board(Board& board);
void print_board(Board& board, bool tidy);
// void py_print_board(Board& board);
int eval_board(Board& board, bool white_to_play);
total_legal_moves_struct total_legal_moves(Board& board, bool white_to_play);
generated_moves_struct generate_moves(Board& board, bool white_to_play);
verified_move verify_move(Board& board, bool white_to_play, int start_sq,
    int dest_sq);
verified_move verify_move(Board& board, bool white_to_play,
    std::string move_letters);
void make_move(Board& board, int start_sq, int dest_sq, int move_modifier);
std::string get_game_outcome(Board& board, bool white_to_play);
int square_letters_to_numbers(std::string square);
std::string square_numbers_to_letters(int square);
bool is_promotion(Board& board, bool white_to_play, std::string move_letters);

#endif