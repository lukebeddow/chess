#include "tree_functions.h"
#include "board_functions.h"

int main(int argc, char** argv)
{
	Game new_game;
	bool human_first;

	char answer;
	std::cout << "Would you like to go first y/n?\n>>>  ";
	while (std::cin >> answer) {

		std::cout << "You entered: " << answer << '\n';

		if (answer == 'y' or answer == 'Y') {
			std::cout << "You have selected to go first (play white)\n";
			human_first = true;
			break;
		}
		else if (answer == 'n' or answer == 'N') {
			std::cout << "You have selected to go second (play black)\n";
			human_first = false;
			break;
		}
		else if (answer == 'q' or answer == 'Q') {
			std::cout << "Quiting application now\n";
			return 0;
		}
		else {
			std::cout << "Valid character not entered, press q to quit. Choose 'y' to play first or 'n' to play second\n";
		}
	}

	char answer2;
	std::cout << "Should engine use neural network evaluator y/n?\n>>>  ";
	while (std::cin >> answer2) {

		std::cout << "You entered: " << answer2 << '\n';

		if (answer2 == 'y' or answer2 == 'Y') {
			std::cout << "You have selected to play against a neural network engine\n";
			new_game.use_nn_evaluator(); // load from the default path
			break;
		}
		else if (answer2 == 'n' or answer2 == 'N') {
			std::cout << "You have selected to play against a traditional engine\n";
			break;
		}
		else if (answer2 == 'q' or answer2 == 'Q') {
			std::cout << "Quiting application now\n";
			return 0;
		}
		else {
			std::cout << "Valid character not entered, press q to quit. Choose 'y' to play first or 'n' to play second\n";
		}
	}

	new_game.play_terminal(human_first);
}