#include "tree_functions.h"
#include "board_functions.h"

int main(int argc, char** argv)
{
    int depth = 6;
    int width = 5;

    // set a specific board state?
    // print the starting board state?
    Board board = create_board();
    bool white_to_play = true;
    bool engine_is_white = true;
    std::cout << "The starting board state is:\n";
    print_board(board, true);

    // make the engine. For testing, we will only do one move
    std::unique_ptr<Engine> engine_pointer = std::make_unique<Engine>();
    engine_pointer->set_depth(depth);
    engine_pointer->set_width(width);

    Move next_move = engine_pointer->generate(board, engine_is_white);
    std::cout << "The move to be played next is " << next_move.to_letters() << '\n';
    make_move(board, next_move.start_sq, next_move.dest_sq, next_move.move_mod);
    std::cout << "The board state after the engine move is:\n";
    print_board(board, true);
}