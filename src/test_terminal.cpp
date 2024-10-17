#include "tree_functions.h"
#include "board_functions.h"

int main(int argc, char** argv)
{
    int depth = 8;
    int width = 5;
    double target_time = 10;
    bool white_to_play;
    bool engine_is_white;
    bool wtp;
    std::string initial_pos;
    Board board;

    // // mate in 1 position
    // initial_pos = "4r2k/1p3rbp/2p1N1p1/p3n3/P2NB1nq/1P6/4R1P1/B1Q2RK1 b - - 4 32"; wtp = false;
    // initial_pos = "4rb2/3qrk2/1p1p1n2/7p/P2P4/4R2P/1BQN1P2/1K4R1 w - - 3 39"; wtp = true;

    // mate in 2 position
    // initial_pos = "4r3/1pp2rbk/6pn/4n3/P3BN1q/1PB2bPP/8/2Q1RRK1 b - - 0 31"; wtp = false;
    // initial_pos = "r2k1b1r/p1ppq2p/np3np1/5p2/1PPP4/P3PQ2/3N1PPP/R1B1K2R w KQ - 1 13"; wtp = true;

    // // mate in 3 position
    // initial_pos = "4r1k1/4r1p1/8/p2R1P1K/5P1P/1QP3q1/1P6/3R4 b - - 0 1"; wtp = false;
    // initial_pos = "r1bqk2r/2ppb1p1/n3P2p/8/2B1nP2/4P3/1PPP3P/RNBQK1NR w KQkq - 0 10"; wtp = true;
    
    // // mate in 4 position
    // initial_pos = "3qr2k/1p3rbp/2p3p1/p7/P2pBNn1/1P3n2/6P1/B1Q1RR1K b - - 1 30"; wtp = false;
    // initial_pos = "r1bqk1nr/pp1p2bp/4n3/2p1Npp1/5P2/2N1P1PP/PPP5/1RBQKB1R w Kkq - 0 10"; wtp = true;

    // position with obvious capture
    initial_pos = "r3r1k1/p1p3b1/2p4p/2q1P3/2P2pb1/1P3N2/P1QN2P1/3RR2K b"; wtp = false;

    // // endgame position
    // initial_pos = "8/3N4/7p/1k4p1/2n3P1/8/5P1P/6K1 b - g3"; wtp = false;

    // set a specific board state?
    // print the starting board state?
    if (initial_pos == "") {
        board = create_board();
        white_to_play = true;
        engine_is_white = true;
    }
    else {
        board = FEN_to_board(initial_pos);
        white_to_play = wtp;
        engine_is_white = wtp;
    }
    
    std::cout << "The starting board state is:\n";
    print_board(board, false);

    // make the engine. For testing, we will only do one move
    std::unique_ptr<Engine> engine_pointer = std::make_unique<Engine>();
    engine_pointer->set_depth(depth);
    engine_pointer->set_width(width);

    Move next_move = engine_pointer->generate(board, engine_is_white, target_time);
    std::cout << "The move to be played next is " << next_move.to_letters() << '\n';
    make_move(board, next_move.start_sq, next_move.dest_sq, next_move.move_mod);
    std::cout << "The board state after the engine move is:\n";
    print_board(board, false);
}