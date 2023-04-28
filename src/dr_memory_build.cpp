#include "tree_functions.h"
#include "board_functions.h"

int main(int argc, char** argv)
{
	// problem move set
	std::vector<std::string> moves{ "e2e4", "h7h5", "g1f3", "g8f6", "b1c3", "a7a5", "d2d4", "b7b6", "f1b5", 
		"c8b7", "d1e2", "b7c8", "e1g1", "f6g8", "f1e1", "c7c5", "d4d5", "c8b7", "c3a4", "d8c7", "a2a3", 
		"a8a7", "b2b3", "g7g6", "g2g3", "a7a8", "c1f4", "c7d8", "c2c3", "f8g7", "a1c1", "f7f5", "f4e5", 
		"g7e5", "f3e5", "g8f6", "e5g6" };

	Game tf_game;
	Board board = create_board(moves);
	bool white_to_play = false;

	board = tf_game.engine_move(board, white_to_play);
}