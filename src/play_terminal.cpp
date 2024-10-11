#include "tree_functions.h"
#include "board_functions.h"

int main(int argc, char** argv)
{
	Game new_game;
	bool human_first;

	// char answer;
	// std::cout << "Would you like to go first y/n?\n>>>  ";
	// while (std::cin >> answer) {

	// 	std::cout << "You entered: " << answer << '\n';

	// 	if (answer == 'y' or answer == 'Y') {
	// 		std::cout << "You have selected to go first (play white)\n";
	// 		human_first = true;
	// 		break;
	// 	}
	// 	else if (answer == 'n' or answer == 'N') {
	// 		std::cout << "You have selected to go second (play black)\n";
	// 		human_first = false;
	// 		break;
	// 	}
	// 	else if (answer == 'q' or answer == 'Q') {
	// 		std::cout << "Quiting application now\n";
	// 		return 0;
	// 	}
	// 	else {
	// 		std::cout << "Valid character not entered, press q to quit. Choose 'y' to play first or 'n' to play second\n";
	// 	}
	// }

	std::cout << "Testing: the engine is set to go first\n";
	human_first = false;

	new_game.play_terminal(human_first);
}