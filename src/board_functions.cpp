#include "board_functions.h"

#if defined(LUKE_PYBIND)
    using namespace pybind11::literals;

    void py_print_board(Board& board, bool neat) {
        /* This function prints the board using the pybind python print interface,
        hopefully allowing it to print in any python command line */

        // initialise variables
        std::string to_print = "";
        std::string print_string;
        int ind;
        int lim;

        // will we print a neat board, or include index information?
        if (neat) lim = 8;
        else      lim = 9;

        // print a board header

        // std::cout << "-----------------------------------------------------------\n";
        py::print("-----------------------------------------------------------");

        // loop through each row on the board
        for (int i = -1; i < lim; ++i) {

            // new line
            if (i == -1)        print_string = "\t";
            else if (i == 0)    print_string = "\n" + std::to_string(8 - i) + "\t";
            //else if (i == 6)    print_string = "\n\t";
            else if (i == 8)    print_string = "\n\t";
            else                print_string = std::to_string(8 - i) + "\t";

            // loop through each column
            for (int j = 0; j < lim; ++j) {

                // are we printing the header?
                if (i == -1) {
                    if (j == 0) to_print = " A ";
                    else if (j == 1) to_print = " B ";
                    else if (j == 2) to_print = " C ";
                    else if (j == 3) to_print = " D ";
                    else if (j == 4) to_print = " E ";
                    else if (j == 5) to_print = " F ";
                    else if (j == 6) to_print = " G ";
                    else if (j == 7) to_print = " H ";
                    else             to_print = "";
                }
                // are we printing an extra bottom row for indexes
                else if (i == 8) {
                    if (j == 0) to_print = " 8 ";
                    else if (j == 1) to_print = " 7 ";
                    else if (j == 2) to_print = " 6 ";
                    else if (j == 3) to_print = " 5 ";
                    else if (j == 4) to_print = " 4 ";
                    else if (j == 5) to_print = " 3 ";
                    else if (j == 6) to_print = " 2 ";
                    else if (j == 7) to_print = " 1 ";
                    else             to_print = "";
                }
                // are we printing indexes at the end of lines
                else if (j == 8) {
                    if (i == 0) to_print = "\t90 ";
                    else if (i == 1) to_print = "\t80 ";
                    else if (i == 2) to_print = "\t70 ";
                    else if (i == 3) to_print = "\t60 ";
                    else if (i == 4) to_print = "\t50 ";
                    else if (i == 5) to_print = "\t40 ";
                    else if (i == 6) to_print = "\t30 ";
                    else if (i == 7) to_print = "\t20 ";
                }
                else { // we are printing the regular board

                    // which board index are we at
                    ind = (((9 - i) * 10) + (8 - j));

                    // what is happening on this square?
                    if (board.arr[ind] == -1)  to_print = "bP ";
                    else if (board.arr[ind] == -2)  to_print = "bN ";
                    else if (board.arr[ind] == -3)  to_print = "bB ";
                    else if (board.arr[ind] == -4)  to_print = "bR ";
                    else if (board.arr[ind] == -5)  to_print = "bQ ";
                    else if (board.arr[ind] == -6)  to_print = "bK ";
                    else if (board.arr[ind] == 1)   to_print = "wP ";
                    else if (board.arr[ind] == 2)   to_print = "wN ";
                    else if (board.arr[ind] == 3)   to_print = "wB ";
                    else if (board.arr[ind] == 4)   to_print = "wR ";
                    else if (board.arr[ind] == 5)   to_print = "wQ ";
                    else if (board.arr[ind] == 6)   to_print = "wK ";
                    else if (board.arr[ind] == 0)   to_print = " . ";
                    else                            to_print = " x ";

                    // for testing! lets visualise the en passant booleans
                    if (ind / 10 == 4) {
                        for (int k = 0; k < 8; ++k) {
                            int en_pass_bool = 53 - ind;
                            if (board.arr[en_pass_bool] == 7) {
                                to_print = " , ";
                            }
                        }
                    }
                    if (ind / 10 == 7) {
                        for (int k = 0; k < 8; ++k) {
                            int en_pass_bool = 91 - ind;
                            if (board.arr[en_pass_bool] == 7) {
                                to_print = " , ";
                            }
                        }
                    }

                }

                print_string += to_print;
                print_string += "  ";

                // print it out
                //py::print(to_print, "  ");
                // std::cout << to_print << "  ";
            }

            py::print(print_string + "\n");
        }

        // include castle rights if doing full print
        if (neat == false) {
            print_string = "\n\t Castle rights (if any): ";
            //std::cout << "\n\n\t Castle rights (if any): ";
            if (board.arr[0] == 7) print_string += "wKS, "; // std::cout << "wKS, ";
            if (board.arr[1] == 7) print_string += "wQS, "; // std::cout << "wQS, ";
            if (board.arr[2] == 7) print_string += "bKS, "; // std::cout << "bKS, ";
            if (board.arr[3] == 7) print_string += "bQS, "; // std::cout << "bQS, ";
            py::print(print_string);
        }

        // print a board header at bottom
        py::print("-----------------------------------------------------------\n");
        // std::cout << "-----------------------------------------------------------\n";

        //// add a double newline at the end
        //std::cout << "\n\n";
    }

    void py_print_board(Board& board)
    {
        /* overload */

        py_print_board(board, false);
    }
#endif

struct evaluation_settings {
    /* This is a global structure that contains settings to determine
    how boards are evaluated */

    int null_defend = 1000;
    int null_attack = 1000;
};

// create global evaluation settings variable
evaluation_settings EVALUATION_SETTINGS;

void set_evaluation_settings(int null_defend, int null_attack) {
    /* This function modifies the global evaluation settings */

    EVALUATION_SETTINGS.null_defend = null_defend;
    EVALUATION_SETTINGS.null_attack = null_attack;
}

piece_moves_struct how_piece_moves(int piece_type, int piece_colour) {
    /*This function fills up a piece_moves structure detailing
    how a piece can move (direction and depth)*/

    // create instance of structure to fill with data
    piece_moves_struct piece_moves;

    // go through each type of piece and assign values
    if (piece_type == 0) throw std::invalid_argument("Piece type = 0");
    // if the piece is a pawn
    else if (piece_type == 1) {
        // is it a white pawn
        if (piece_colour == 1) {
            piece_moves.directions = { -9, -10, -11 };
            piece_moves.depth = 1;
        }
        // is it a black pawn
        else if (piece_colour == 2) {
            piece_moves.directions = { 9, 10, 11 };
            piece_moves.depth = 1;
        }
    }
    // if the piece is a knight
    else if (piece_type == 2) {
        piece_moves.directions = { -21, -19, -12, -8, 8, 12, 19, 21 };
        piece_moves.depth = 1;
    }
    // if the piece is a bishop
    else if (piece_type == 3) {
        piece_moves.directions = { -11, -9, 9, 11 };
        piece_moves.depth = 7;
    }
    // if the piece is a rook
    else if (piece_type == 4) {
        piece_moves.directions = { -10, -1, 1, 10 };
        piece_moves.depth = 7;
    }
    // if the piece is a queen
    else if (piece_type == 5) {
        piece_moves.directions = { -11, -10, -9, -1, 1, 9, 10, 11 };
        piece_moves.depth = 7;
    }
    // if the piece is a king
    else if (piece_type == 6) {
        piece_moves.directions = { -11, -10, -9, -1, 1, 9, 10, 11 };
        piece_moves.depth = 1;
    }
    // else bad input
    else throw std::invalid_argument("Piece type was not between 1 and 6");

    return piece_moves;
}

int square_letters_to_numbers(std::string square)
{
    /* convert a square representation in letters eg A3 to numbers eg 24 */

    if (square.size() != 2) {
        throw std::runtime_error("square should be a string of two letters eg 'a2'");
    }
    else if (square[0] >= 65 and square[0] <= 72) {
        // then it uses a capital letter, convert to lower case
        square[0] += 32;
    }
    else if (square[0] < 97 or square[0] > 104 or
        square[1] < 49 or square[1] > 56) {
        throw std::runtime_error("square needs to be a-h and 1-8 eg 'e4'");
    }

    std::string num_string;

    char y = square[0];
    char x = square[1];

    num_string += x + 1; // '1' becomes '2' etc
    num_string += 153 - y; // 'h' becomes '1' etc

    int square_number = std::stoi(num_string);

    return square_number;
}

std::string square_numbers_to_letters(int square)
{
    /* convert square format as numbers eg 24 to letters eg a3, always lower case */

    // create strings
    std::string move_letters;
    std::string ind_letters = std::to_string(square);

    if (ind_letters.size() != 2) {
        throw std::runtime_error("square string not 2 digits");
    }

    // what are the two numbers that make up this square
    char x = ind_letters[0];
    char y = ind_letters[1];

    move_letters += 153 - y; // '1' becomes 'h' etc
    move_letters += x - 1; // '2' becomes '1' etc

    // catch any errors in the other function
    int not_used = square_letters_to_numbers(move_letters);

    return move_letters;
}

Board create_board() {
    /*This recieves a board array and sets everything to its default values*/

    //static int board[120];
    Board board;

    // loop through and set up the pieces
    for (int i = 0; i < 120; ++i) {

        // default
        board.arr[i] = -7;

        // set castle rights booleans to true
        if (i == 0 or i == 1 or i == 2 or i == 3) board.arr[i] = 7;

        // white pawns
        if ((i / 10 == 3) and (i != 30) and (i != 39)) {
            board.arr[i] = 1;
        }
        // black pawns
        if ((i / 10 == 8) and (i != 80) and (i != 89)) {
            board.arr[i] = -1;
        }

        // white pieces
        if (i == 21 or i == 28) board.arr[i] = 4;       // rooks
        if (i == 22 or i == 27) board.arr[i] = 2;       // knights
        if (i == 23 or i == 26) board.arr[i] = 3;       // bishops
        if (i == 25) board.arr[i] = 5;                  // queen
        if (i == 24) board.arr[i] = 6;                  // king

        // black pieces
        if (i == 91 or i == 98) board.arr[i] = -4;      // rooks
        if (i == 92 or i == 97) board.arr[i] = -2;      // knights
        if (i == 93 or i == 96) board.arr[i] = -3;      // bishops
        if (i == 95) board.arr[i] = -5;                 // queen
        if (i == 94) board.arr[i] = -6;                 // king

        // board squares
        if (i > 40 and i < 49) board.arr[i] = 0;
        if (i > 50 and i < 59) board.arr[i] = 0;
        if (i > 60 and i < 69) board.arr[i] = 0;
        if (i > 70 and i < 79) board.arr[i] = 0;

    }

    return board;
}

bool check_board(Board& board)
{
    /* this function checks that a board falls within normal values */

    for (int a : board.arr) {
        if (a > 7 or a < -7) {
            std::cout << "Board is corrupted!\n";
            return false;
        }
    }

    return true;
}

// TESTING declare early, no definition
int is_passed_pawn(Board& board, int our_colour, int square_num);

void print_board(Board& board, bool neat) {
    /*This function prints the board on the command line*/

    // initialise variables
    std::string to_print = "";
    int ind;
    int lim;

    // will we print a neat board, or include index information?
    if (neat) lim = 8;
    else      lim = 9;

    // print a board header
    std::cout << "-----------------------------------------------------------\n";

    // loop through each row on the board
    for (int i = -1; i < lim; ++i) {

        // new line
        if (i == -1) std::cout << '\t';
        else if (i == 8)  std::cout << "\n\n\n\t";
        else              std::cout << "\n\n" << (8 - i) << '\t';

        // loop through each column
        for (int j = 0; j < lim; ++j) {

            // are we printing the header?
            if (i == -1) {
                if (j == 0) to_print = " A ";
                else if (j == 1) to_print = " B ";
                else if (j == 2) to_print = " C ";
                else if (j == 3) to_print = " D ";
                else if (j == 4) to_print = " E ";
                else if (j == 5) to_print = " F ";
                else if (j == 6) to_print = " G ";
                else if (j == 7) to_print = " H ";
                else             to_print = "";
            }
            // are we printing an extra bottom row for indexes
            else if (i == 8) {
                if (j == 0) to_print = " 8 ";
                else if (j == 1) to_print = " 7 ";
                else if (j == 2) to_print = " 6 ";
                else if (j == 3) to_print = " 5 ";
                else if (j == 4) to_print = " 4 ";
                else if (j == 5) to_print = " 3 ";
                else if (j == 6) to_print = " 2 ";
                else if (j == 7) to_print = " 1 ";
                else             to_print = "";
            }
            // are we printing indexes at the end of lines
            else if (j == 8) {
                if (i == 0) to_print = "\t90 ";
                else if (i == 1) to_print = "\t80 ";
                else if (i == 2) to_print = "\t70 ";
                else if (i == 3) to_print = "\t60 ";
                else if (i == 4) to_print = "\t50 ";
                else if (i == 5) to_print = "\t40 ";
                else if (i == 6) to_print = "\t30 ";
                else if (i == 7) to_print = "\t20 ";
            }
            else { // we are printing the regular board

                // which board index are we at
                ind = (((9 - i) * 10) + (8 - j));

                // what is happening on this square?
                if (board.arr[ind] == -1)  to_print = "bP ";
                else if (board.arr[ind] == -2)  to_print = "bN ";
                else if (board.arr[ind] == -3)  to_print = "bB ";
                else if (board.arr[ind] == -4)  to_print = "bR ";
                else if (board.arr[ind] == -5)  to_print = "bQ ";
                else if (board.arr[ind] == -6)  to_print = "bK ";
                else if (board.arr[ind] == 1)   to_print = "wP ";
                else if (board.arr[ind] == 2)   to_print = "wN ";
                else if (board.arr[ind] == 3)   to_print = "wB ";
                else if (board.arr[ind] == 4)   to_print = "wR ";
                else if (board.arr[ind] == 5)   to_print = "wQ ";
                else if (board.arr[ind] == 6)   to_print = "wK ";
                else if (board.arr[ind] == 0)   to_print = " . ";
                else                            to_print = " x ";

                //// for testing! lets visualise passed pawn scores (0=not passed...)
                //if (board.arr[ind] == 1 or board.arr[ind] == -1) {
                //    int passed = is_passed_pawn(board, board.arr[ind], ind);
                //    std::string passed_str = std::to_string(passed);
                //    to_print += passed_str;
                //}

                // for testing! lets visualise the en passant booleans
                if (ind / 10 == 4) {
                    for (int k = 0; k < 8; ++k) {
                        int en_pass_bool = 53 - ind;
                        if (board.arr[en_pass_bool] == 7) {
                            to_print = " , ";
                        }
                    }
                }
                if (ind / 10 == 7) {
                    for (int k = 0; k < 8; ++k) {
                        int en_pass_bool = 91 - ind;
                        if (board.arr[en_pass_bool] == 7) {
                            to_print = " , ";
                        }
                    }
                }

            }

            // print it out
            std::cout << to_print << "  ";
        }
    }

    // include castle rights if doing full print
    if (neat == false) {
        std::cout << "\n\n\t Castle rights (if any): ";
        if (board.arr[0] == 7) std::cout << "wKS, ";
        if (board.arr[1] == 7) std::cout << "wQS, ";
        if (board.arr[2] == 7) std::cout << "bKS, ";
        if (board.arr[3] == 7) std::cout << "bQS, ";
    }

    // print a board header at bottom
    std::cout << "\n";
    std::cout << "-----------------------------------------------------------\n";

    //// add a double newline at the end
    //std::cout << "\n\n";
}

void print_board(Board& board)
{
    /* overload */
    print_board(board, false);
}

bool is_in_check(Board& board, int check_square, int piece_colour) {
    /*This function determines if a square is under attack by the opposite
    colour to the colour given*/

    // if the specified square is out of bounds
    if (board.arr[check_square] == 7 or board.arr[check_square] == -7)
        throw std::invalid_argument("Square given is out of bounds");

    int piece_one;                  // 1st kind of piece
    int piece_two;                  // 2nd kind of piece
    int dest_sq;                    // destination square
    int attack_sign;                // sign of pieces that attack us

    // what colour are we, use this to set who our attackers are
    if (piece_colour == 1) attack_sign = -1;
    else if (piece_colour == -1) attack_sign = 1;
    else throw std::invalid_argument("is_in_check: Piece colour must be 1 (white) or -1 (black)");

    // four pieces that account for all possible movements
    int move_group_one[4] = { 6, 2, 3, 4 };
    int move_group_two[4] = { 1, 2, 5, 5 };

    // for each of the four movements
    for (int i = 0; i < 4; ++i) {

        // how do these pieces move
        piece_moves_struct how_moves = how_piece_moves(move_group_one[i], piece_colour);

        // what pieces can attack us in these directions
        piece_one = move_group_one[i] * attack_sign;
        piece_two = move_group_two[i] * attack_sign;

        // check all the directions these pieces move in
        for (int move_dist : how_moves.directions) {

            // arithmatic begins at starting square
            dest_sq = check_square;

            // check all the squares these pieces can reach
            for (int j = 0; j < how_moves.depth; ++j) {

                dest_sq += move_dist;

                // is the square out of bounds
                if (board.arr[dest_sq] == -7 or board.arr[dest_sq] == 7)
                    break;

                // if the square contains an opposing piece of this type
                if (board.arr[dest_sq] == piece_one or board.arr[dest_sq] == piece_two) {

                    // check if the piece is a pawn (different rules apply)
                    if (board.arr[dest_sq] == 1 * attack_sign) {
                        // can the pawn attack us
                        if (dest_sq != check_square - (9 * attack_sign) and
                            dest_sq != check_square - (11 * attack_sign))
                            continue; // if it cannot attack us
                    }
                    // hence, the piece must attack the square
                    return true;
                }
                // else does the square contain a blocking piece
                else if (board.arr[dest_sq] != 0)
                    break; // since no more checks can occur along this line
            }
        }
    }

    // finally, we need to check if a double pawn move can check this square
    // currently, we do not check this since this function is only called to
    // check whether castling is legal

    // if we get here, the square must not be in check
    return false;
}

bool list_of_castles(Board& board, int piece_colour, 
    std::vector<int>& castle_list) {
    /* This function fills a vector with any allowed castling moves */

    int i = 0;                      // indexing variable
    int rank = 0;                   // rank at which castling occurs
    int king_sq;                    // king start sq for a legal castle
    bool any_castles = false;       // are there any legal castling moves

    // check if this player has castle rights
    if (piece_colour == 1) {
        // if both castle booleans are set false (-7)
        if (board.arr[0] == -7 and board.arr[1] == -7)
            return false;
        // define some castling information for white
        i = 0;
        rank = 20;
        king_sq = 24;
    }
    else if (piece_colour == -1) {
        // if both castle booleans are set false (-7)
        if (board.arr[2] == -7 and board.arr[3] == -7)
            return false;
        // define some castling information for black
        i = 2;
        rank = 90;
        king_sq = 94;
    }
    else throw std::invalid_argument("list_of_castles: Piece colour can only be 1 (white) or -1 (black)");

    // define what castling is (we will add 20 to these for white, 90 for black)
    // first half of arrays below are for kingside, next half for queenside
    int castle_destination[2] = { 2 + rank, 6 + rank };
    int squares_no_check[6] = { 2 + rank, 3 + rank, 4 + rank, 4 + rank, 5 + rank, 6 + rank };
    int squares_need_empty[6] = { 2 + rank, 3 + rank, 3 + rank, 5 + rank, 6 + rank, 7 + rank };

    int rook_squares[2] = { 1 + rank, 8 + rank };

    // loop through kingside and queenside
    for (int j = 0; j < 2; ++j) {

        // if the player has the right to castle on this side
        if (board.arr[i + j] == 7) {

            // check that all required squares are empty
            if (board.arr[squares_need_empty[0 + (j * 3)]] == 0 and
                board.arr[squares_need_empty[1 + (j * 3)]] == 0 and
                board.arr[squares_need_empty[2 + (j * 3)]] == 0) {

                // check if that none of the three castling squares are in check
                if (not (is_in_check(board, squares_no_check[0 + (j * 3)], piece_colour) or
                    is_in_check(board, squares_no_check[1 + (j * 3)], piece_colour) or
                    is_in_check(board, squares_no_check[2 + (j * 3)], piece_colour))) {

                    // check we have a rook to move
                    if (board.arr[rook_squares[j]] == 4 or
                        board.arr[rook_squares[j]] == -4) {

                        // hence, the castling move is legal, save it
                        castle_list.push_back(king_sq);
                        castle_list.push_back(castle_destination[j]);
                        castle_list.push_back(5);   // move modifier = 5

                        // we have at least one legal castle
                        any_castles = true;
                    }

                    
                }
            }
        }
    }

    return any_castles;
}

void move_piece_on_board(Board& board, int start_sq, int dest_sq) {
    /*This function moves a piece from a starting square to a destination
    square. This function detects and manages en passant booleans needing
    to be set to true or all wiped to false*/

    // move the piece
    board.arr[dest_sq] = board.arr[start_sq];
    board.arr[start_sq] = 0;

    // if the passant wipe boolean is set true
    // NB: for the board array -> 7 = true, -7 = false
    if (board.arr[4] == 7) {
        // wipe all the en passant booleans, set all to false
        for (int i = 0; i < 16; ++i) {
            // passant booleans are from board[5] to board[20] inclusive
            board.arr[i + 5] = -7;
        }

        // now we have reset them, set passant wipe boolean to false
        board.arr[4] = -7; // 7 = true, -7 = false
    }

    // was this move a pawn going two squares
    if ((board.arr[dest_sq] == 1 or board.arr[dest_sq] == -1) and
        (dest_sq - start_sq == 20 or dest_sq - start_sq == -20)) {

        // if it is a white pawn
        if (start_sq / 10 == 3) {
            int en_pass_square = start_sq + 10;
            int en_pass_bool = 53 - en_pass_square;
            board.arr[en_pass_bool] = 7; // 7 = true, -7 = false
        }
        // if it is a black pawn
        else if (start_sq / 10 == 8) {
            int en_pass_square = start_sq - 10;
            int en_pass_bool = 91 - en_pass_square;
            board.arr[en_pass_bool] = 7; // 7 = true, -7 = false
        }

        // now set the passant wipe boolean to true so it is wiped next move
        board.arr[4] = 7; // 7 = true, -7 = false
    }

    // check if this move removed castle rights

    // what colour of piece just moved
    if (board.arr[dest_sq] > 0) {
        // white moved, does white have castle rights?
        if (board.arr[0] == 7 or board.arr[1] == 7) {
            // did a king just move
            if (board.arr[dest_sq] == 6) {
                // any king movement removes all castle rights
                board.arr[0] = -7;
                board.arr[1] = -7;
            }
            // did a rook just move
            else if (board.arr[dest_sq] == 4) {
                // was it the kingside rook
                if (start_sq == 21) {
                    // remove kingside castling rights
                    board.arr[0] = -7;
                }
                // was it the queenside rook
                else if (start_sq == 28) {
                    // remove queenside castling rights
                    board.arr[1] = -7;
                }
            }
        }
    }
    else if (board.arr[dest_sq] < 0) {
        // black moved, does black have castle rights?
        if (board.arr[2] == 7 or board.arr[3] == 7) {
            // did a king just move
            if (board.arr[dest_sq] == -6) {
                // any king movement removes all castle rights
                board.arr[2] = -7;
                board.arr[3] = -7;
            }
            // did a rook just move
            else if (board.arr[dest_sq] == -4) {
                // was it the kingside rook
                if (start_sq == 91) {
                    // remove kingside castling rights
                    board.arr[2] = -7;
                }
                // was it the queenside rook
                else if (start_sq == 98) {
                    // remove queenside castling rights
                    board.arr[3] = -7;
                }
            }
        }
    }
    // if board.arr[dest_sq] == 0, we haven't moved anything
    else {
        std::cout << "The movement asked for was " << start_sq << ", " << dest_sq << '\n';
        throw std::runtime_error("move_piece_on_board has failed"); }
}

void make_move(Board& board, int start_sq, int dest_sq, int move_modifier) {
    /*This function makes a chess move. All the internals such as castle rights
    being removed or en passant booleans are handled internally. However, this
    function does not check if a move is legal*/

    // make the move on the board
    move_piece_on_board(board, start_sq, dest_sq);

    /* now check the move modifier to account for additional effects
    1 = move to empty square
    2 = capture
    3 = capture en passant
    4 = NONE (good to change this later, after finishing python testing)
    5 = castle
    6 = promotion to knight
    7 = promotion to queen
    8 = promotion to bishop
    9 = promotion to rook       */

    // if the move was to an empty square
    if (move_modifier == 1) { /* do nothing */ }
    
    // if the move was a capture
    else if (move_modifier == 2) { /* do nothing */ }

    // if the move was a capture en passant
    else if (move_modifier == 3) {
        // was the capturing pawn white
        if (board.arr[dest_sq] == 1) {
            // remove the pawn behind it
            board.arr[dest_sq - 10] = 0;
        }
        // else if the capturing pawn was black
        else if (board.arr[dest_sq] == -1) {
            // remove the pawn in front of it
            board.arr[dest_sq + 10] = 0;
        }
        else { throw std::runtime_error("make_move en passant logic failed"); }
    }

    // if the move was a castle
    else if (move_modifier == 5) {
        // what kind of castle was it? We need to also move the rook
        if      (dest_sq == 22) { move_piece_on_board(board, 21, 23); }
        else if (dest_sq == 26) { move_piece_on_board(board, 28, 25); }
        else if (dest_sq == 92) { move_piece_on_board(board, 91, 93); }
        else if (dest_sq == 96) { move_piece_on_board(board, 98, 95); }
        // save that this player has castled
        if (board.arr[dest_sq] > 0) {
            // set the white has castled boolean to true
            board.arr[101] = 7;
        }
        else if (board.arr[dest_sq] < 0) {
            // set the black has castled boolean to true
            board.arr[102] = 7;
        }
        else {
            throw std::runtime_error("castling logic failed, dest sq is empty");
        }
    }

    // if the piece is promoting to a knight
    else if (move_modifier == 6) {
        // change pawn (+1 or -1) to a knight of the same colour (+2 or -2)
        board.arr[dest_sq] *= 2;
        // check that it worked
        if (abs(board.arr[dest_sq]) != 2) {
            throw std::runtime_error("make_move has been asked to knight promote \
                                        a piece that isn't a pawn");
            
        }
    }

    // if the piece is promoting to a queen
    else if (move_modifier == 7) {
        // change pawn (+1 or -1) to a queen of the same colour (+5 or -5)
        board.arr[dest_sq] *= 5;
        // check that it worked
        if (abs(board.arr[dest_sq]) != 5) {
            throw std::runtime_error("make_move has been asked to queen promote \
                                        a piece that isn't a pawn");
        }
    }

    // if the piece is promoting to a bishop
    else if (move_modifier == 8) {
        // change pawn (+1 or -1) to a bishop of the same colour (+3 or -3)
        board.arr[dest_sq] *= 3;
        // check that it worked
        if (abs(board.arr[dest_sq]) != 3) {
            throw std::runtime_error("make_move has been asked to bishop promote \
                                        a piece that isn't a pawn");
        }
    }

    // if the piece is promoting to a rook
    else if (move_modifier == 9) {
        // change pawn (+1 or -1) to a rook of the same colour (+4 or -4)
        board.arr[dest_sq] *= 4;
        // check that it worked
        if (abs(board.arr[dest_sq]) != 4) {
            throw std::runtime_error("make_move has been asked to rook promote \
                                        a piece that isn't a pawn");
        }
    }

    // else we did not recive a move modifier that we expected
    else {
        throw std::invalid_argument("make_move has recieved an incorrect \
                                            move modifier (not 1 <= int x <= 9");
    }
}

piece_attack_defend_struct piece_attack_defend(Board& board, int square_num,
    int our_piece, int our_colour) {
    /* This function looks at a single square, and finds all the pieces that
    attack this square and defend it; as well as the pieces attacked from
    square and all the pieces in view of this square*/

    // check we have correct inputs
    if (our_piece == 0) {
        throw std::invalid_argument("piece_attack_defend has been given our_piece = 0, ie empty square");
    }
    else if (our_piece == 7 or our_piece == -7) {
        throw std::invalid_argument("piece_attack_defend has been given our_piece = 7, ie non legal square");
    }
    else if (our_piece < 0) { 
        // ensure our piece is a +ve number
        our_piece *= -1;
    }

    // four pairs of pieces that account for all possible movements
    int move_group_one[4] = { 6, 2, 3, 4 };
    int move_group_two[4] = { 1, 2, 5, 5 };

    // initialise structures
    piece_moves_struct movements;               // store how pieces move
    piece_attack_defend_struct output;          // store output of this function

    // initialise variables
    bool pawn = false;          // is this piece a pawn
    bool first_loop = true;     // are we on the first loop
    bool hostile_block;         // is a hostile piece blocking along this line
    bool hostile_attacker;      // is a hostile piece attacking us along this line
    bool friendly_block;        // is a friendly piece blocking along this line
    bool friendly_defender;     // is a friendly piece defending us along this line
    bool second_block;          // have we reached a second blocking piece
    bool ignore_interactions;   // after a second blocker, we ignore piece interactions

    int dest_sq;                // destination square
    int attack;                 // attack value (-1 for 1st attacker, -2 for 2nd etc)
    int defend;                 // defend value (+1 for 1st defender, +2 for 2nd etc)
    int priority;               // our attack value (-1 for 1st piece attacked etc)
    int sq_value;               // value of a square being looked at
    int passant_sq;             // square an en passant attacker will move to
    int our_pass_rank;          // rank which when we are on we can captured en passant
    int their_pass_rank;        // rank which when they are on can be captured en passant
    int our_pass_adj;           // to find passant boolean for a square we want to capture
    int their_pass_adj;         // to find passant boolean for a square they want to capture
    int block_piece;            // value of a blocking piece
    int sign;                   // +1 for white, -1 for black

    int en_passant[2];          // squares from which we could be taken en passant
    int pawn_attacks[2];        // squares from which pawns can attack us
    int pawn_defends[2];        // squares from which pawns can defend us

    // are we a pawn?
    if (our_piece == 1 or our_piece == -1) { pawn = true; }

    // define pawn behaviour
    en_passant[0] = square_num - 1;
    en_passant[1] = square_num + 1;
    if (our_colour == 1) {
        pawn_attacks[0] = square_num + 9;
        pawn_attacks[1] = square_num + 11;
        pawn_defends[0] = square_num - 9;
        pawn_defends[1] = square_num - 11;
        passant_sq = -10;
        our_pass_rank = 4;
        their_pass_rank = 7;
        our_pass_adj = 53;
        their_pass_adj = 91;
        sign = 1;
    }
    else if (our_colour == -1) {
        pawn_attacks[0] = square_num - 9;
        pawn_attacks[1] = square_num - 11;
        pawn_defends[0] = square_num + 9;
        pawn_defends[1] = square_num + 11;
        passant_sq = 10;
        our_pass_rank = 7;
        their_pass_rank = 4;
        our_pass_adj = 91;
        their_pass_adj = 53;
        sign = -1;
    }
    else throw std::invalid_argument("our colour not equal to -1 or +1 in piece_attack_defend");

    constexpr int piece_movements[24] = {
        -11, -10, -9, -1, 1, 9, 10, 11,
        -21, -19, -12, -8, 8, 12, 19, 21,
        -11, -9, 9, 11,
        -10, -1, 1, 10
    };

    // for each of the four pieces that cover all possible movements
    for (int i = 0; i < 4; i++) {

        //// find out how this piece moves
        //movements = how_piece_moves(move_group_one[i], our_colour);

        // testing
        int xx, yy, zz;
        if (i == 0) {
            xx = 0;
            yy = 8;
            zz = 1;
        }
        else if (i == 1) {
            xx = 8;
            yy = 16;
            zz = 1;
        }
        else if (i == 2) {
            xx = 16;
            yy = 20;
            zz = 7;
        }
        else if (i == 3) {
            xx = 20;
            yy = 24;
            zz = 7;
        }

        // check all the directions in which these pieces can move
        //for (int move_dist : movements.directions) {
        for (int a = xx; a < yy; a++) {

            int move_dist = piece_movements[a];

            // arithmetic begins at starting square
            dest_sq = square_num;

            // reset all booleans when we start a new line/direction
            hostile_block = false;
            hostile_attacker = false;
            friendly_block = false;
            friendly_defender = false;
            second_block = false;
            ignore_interactions = false;

            // reset all attack and defend values
            attack = -1;
            defend = 1;
            priority = -1;

            // check all the squares these pieces can reach
            //for (int j = 0; j < movements.depth; j++) {
            for (int j = 0; j < zz; j++) {

                // check the next square along this line
                dest_sq += move_dist;

                // what is on this square (scaled by our colour so +ve == our piece)
                sq_value = board.arr[dest_sq] * sign;

                // is this square on the board? If not, this line is over
                if (sq_value == 7 or sq_value == -7) break;

                // does this square contain a piece
                if (sq_value != 0) {

                    // save the view of the piece
                    if (not first_loop) {
                        output.piece_view.push_back(dest_sq);
                    }

                    // keep checking squares for piece_view
                    if (ignore_interactions) continue;

                    // is the piece the same colour as us
                    if (sq_value > 0) {

                        // it is the same colour, does it defend us
                        if (sq_value == move_group_one[i] or
                            sq_value == move_group_two[i]) {

                            // check if its a pawn, which may not defend us
                            if (first_loop and sq_value == 1) {
                                if (dest_sq != pawn_defends[0] and
                                    dest_sq != pawn_defends[1]) {
                                    continue; // because the pawn is not defending
                                }
                            }

                            // so the piece in the square defends us

                            // check if the sight of this piece is blocked
                            if (friendly_block) {
                                output.defend_me.insert(output.defend_me.end(),
                                    { dest_sq, sq_value, 20 + block_piece });
                            }
                            else if (hostile_block) {
                                output.defend_me.insert(output.defend_me.end(),
                                    { dest_sq, sq_value, 30 + block_piece });
                            }
                            else {
                                output.defend_me.insert(output.defend_me.end(),
                                    { dest_sq, sq_value, defend });

                                // we are behind a friendly piece along this line
                                friendly_defender = true;
                                defend += 1;
                                priority -= 1;
                                block_piece = sq_value;
                            }
                        }
                        // it is our colour but it doesn't defend us
                        else { 

                            // add special behaviour for bishops/queens behind pawns
                            if (sq_value == 1 and move_group_one[i] == 3 and
                                (dest_sq == pawn_defends[0] or
                                    dest_sq == pawn_defends[1])) {

                                // the pawn does not block along this line
                                // (since it attacks diagonally and so do we)
                                defend += 1;
                                friendly_defender = true;
                                block_piece = 1;
                                continue;
                            }

                            // hence this piece blocks along this line

                            // if we have already passed pieces before this
                            if (friendly_defender or hostile_attacker) {
                                // then this is effectively the second blocking piece
                                second_block = true;
                            }

                            // stop at the second blocker
                            // NEW EXPERIMENT
                            //if (second_block) break;
                            // ignore piece interactions after the second blocker
                            if (second_block) {
                                //old_break = true;
                                ignore_interactions = true;
                                continue;
                                //break;
                            }

                            // save that this piece blocks along this line
                            friendly_block = true;
                            second_block = true;
                            block_piece = sq_value;
                            defend = 1;
                        }
                    }
                    // the square contains an opposing piece
                    else { 

                        // convert the square value to a +ve number
                        sq_value *= -1;

                        // check if the piece is attacking us
                        if (sq_value == move_group_one[i] or
                            sq_value == move_group_two[i]) {

                            // check if it's a pawn, which may not attack us
                            if (first_loop and sq_value == 1) {
                                
                                // if we aren't on a square a pawn can attack us from
                                if (dest_sq != pawn_attacks[0] and
                                    dest_sq != pawn_attacks[1]) {
                                    // check to confirm that it cannot capture en passant
                                    // if it can capture us en passant
                                    // we have to be a pawn
                                    // they have to be a pawn
                                    // we have to be on our 4th rank
                                    // they have to be on our 4th rank
                                    // they have to attack the square behind us
                                    // the passant boolean must be true (+7)

                                    // can the pawn attack us en passant
                                    if (pawn and (dest_sq == en_passant[0] or
                                        dest_sq == en_passant[1])
                                        and (dest_sq / 10 == our_pass_rank + sign)) {
                                        if (board.arr[our_pass_adj - (square_num + passant_sq)] == 7) {
                                            // hence the pawn can attack us en passant
                                            output.attack_me.insert(output.attack_me.end(),
                                                { dest_sq, sq_value, attack });
                                            attack -= 1;
                                            hostile_attacker = true;
                                            block_piece = sq_value;
                                        }
                                    }

                                    // can we attack the pawn
                                    if ((our_piece == move_group_one[i] or
                                        our_piece == move_group_two[i])
                                        and not
                                        (pawn and (dest_sq != pawn_attacks[0] and
                                            dest_sq != pawn_attacks[1]))) {
                                        output.attack_list.insert(output.attack_list.end(),
                                            { dest_sq, sq_value, priority });

                                        priority -= 1;
                                    }

                                    // either the pawn cannot attack us, or we already
                                    // accounted for an en passant attack
                                    continue;
                                }
                            }

                            // so the piece in the square attacks us

                            // if we have already passed a friendly blocking piece
                            if (friendly_defender) {
                                friendly_block = true;
                                second_block = true;
                                defend = 1;
                            }

                            // check if the sight of the piece is blocked
                            if (friendly_block) {
                                output.attack_me.insert(output.attack_me.end(),
                                    { dest_sq, sq_value, -20 - block_piece });
                                // check if we can attack the piece
                                if ((our_piece == move_group_one[i] or
                                    our_piece == move_group_two[i])
                                    and not
                                    (pawn and (dest_sq != pawn_attacks[0] and
                                        dest_sq != pawn_attacks[1])))
                                    output.attack_list.insert(output.attack_list.end(),
                                        { dest_sq, sq_value, -20 - block_piece});
                            }
                            else if (hostile_block) {
                                output.attack_me.insert(output.attack_me.end(),
                                    { dest_sq, sq_value, -30 - block_piece });
                                // check if we can attack the piece
                                if ((our_piece == move_group_one[i] or
                                    our_piece == move_group_two[i])
                                    and not
                                    (pawn and (dest_sq != pawn_attacks[0] and
                                        dest_sq != pawn_attacks[1])))
                                    output.attack_list.insert(output.attack_list.end(),
                                        { dest_sq, sq_value, -30 - block_piece });
                            }
                            else { // the sight of the piece is not blocked
                                output.attack_me.insert(output.attack_me.end(),
                                    { dest_sq, sq_value, attack });
                                attack -= 1;
                                hostile_attacker = true;
                                block_piece = sq_value;

                                // check if we can attack the piece
                                if ((our_piece == move_group_one[i] or
                                    our_piece == move_group_two[i])
                                    and not
                                    (pawn and (dest_sq != pawn_attacks[0] and
                                        dest_sq != pawn_attacks[1]))) {
                                    output.attack_list.insert(output.attack_list.end(),
                                        { dest_sq, sq_value, priority });

                                    priority -= 1;
                                }
                            }
                        }
                        // it is an opposing piece but it doesn't attack us
                        else {

                            // add special behaviour for bishops/queens behind pawns
                            if (sq_value == 1 and move_group_one[i] == 3 and
                                (dest_sq == pawn_attacks[0] or
                                    dest_sq == pawn_attacks[1])) {

                                // the pawn does not block along this line
                                // (since it attacks diagonally and so do we)
                                attack -= 1;
                                hostile_attacker = true;
                                block_piece = 1;

                                // can we attack the pawn
                                if ((our_piece == move_group_one[i] or
                                    our_piece == move_group_two[i])
                                    and not
                                    (pawn and (dest_sq != pawn_attacks[0] and
                                        dest_sq != pawn_attacks[1]))) {
                                    output.attack_list.insert(output.attack_list.end(),
                                        { dest_sq, sq_value, priority });

                                    priority -= 1;
                                }
                                continue;
                            }

                            // hence this piece blocks along this line

                            // if we have already passed pieces before this
                            if (friendly_defender or hostile_attacker) {
                                // then this is effectively the second blocking piece
                                second_block = true;
                                // the first piece we passed is now a blocker
                                if (friendly_defender) friendly_block = true;
                                else if (hostile_attacker) hostile_block = true;
                            }

                            // can we attack this piece
                            if ((our_piece == move_group_one[i] or
                                our_piece == move_group_two[i])
                                and not
                                (pawn and (dest_sq != pawn_attacks[0] and
                                    dest_sq != pawn_attacks[1]))) {
                                // is our view blocked
                                if (friendly_block) {
                                    output.attack_list.insert(output.attack_list.end(),
                                        { dest_sq, sq_value, -20 - block_piece });
                                }
                                else if (hostile_block) {
                                    output.attack_list.insert(output.attack_list.end(),
                                        { dest_sq, sq_value, -30 - block_piece });
                                }
                                else { // our sight of the piece is not blocked
                                    output.attack_list.insert(output.attack_list.end(),
                                        { dest_sq, sq_value, priority });
                                }
                            }

                            // stop at the second blocker
                            // NEW EXPERIMENT
                            //if (second_block) break;
                            // ignore piece interactions after the second blocker
                            if (second_block) {
                                ignore_interactions = true;
                                continue;
                                //break;
                            }

                            // save that this piece blocks along this line
                            hostile_block = true;
                            second_block = true;
                            block_piece = sq_value;
                            attack = -1;

                        }
                    }
                }
                // the square does not contain a piece
                else {
                    
                    // NEW EXPERIMENT
                    // keep going until the edge of the board for piece_view
                    if (ignore_interactions) continue;

                    // if there is any blocking at all, don't count this square
                    if (friendly_block or hostile_block or
                        friendly_defender or hostile_attacker) {
                        continue;
                    }

                    // otherwise, the square is in direct view

                    // check if we can attack the square
                    if ((our_piece == move_group_one[i] or
                        our_piece == move_group_two[i])
                        and not
                        (pawn and (dest_sq != pawn_attacks[0] and
                            dest_sq != pawn_attacks[1]))) {

                        // now check if this attack can capture en passant
                        // we have to be a pawn on their 4th rank 
                        // they have to be a pawn on their 4th rank
                        // we have to attack the square behind them
                        // the passant boolean must be true (+7)

                        // check if our attack on this empty square is en passant
                        if (pawn and dest_sq / 10 == their_pass_rank) {
                            // is the passant boolean set to true
                            if (board.arr[their_pass_adj - dest_sq] == 7) {
                                output.attack_list.insert(output.attack_list.end(),
                                    { dest_sq, 1, -1 });
                                continue;
                            }
                        }

                        // else we attack an empty square with no capture threat
                        output.attack_list.insert(output.attack_list.end(),
                            { dest_sq, 0, 0 });
                    }

                    // if our piece is a king, this is a square he can be attacked from
                    // (avoid double counting in the first loop)
                    if (our_piece == 6 and not first_loop) {
                        output.attack_me.insert(output.attack_me.end(),
                            { dest_sq, 0, 0 });
                    }

                }
                // finally, loop to the next move depth
            }
            // we have finished a line
        }
        // we have finished a move group
        first_loop = false;
    }
    // we have finished all all of the move groups
    return output;
}

int find_pins_checks(Board& board, bool& check, bool& pin, int king_index,
    int player_colour, std::vector<int>& block_check, std::vector<int>& pinned_moves,
    std::vector<int>& pinned_piece, piece_attack_defend_struct& king_pad_struct) {
    /* This function finds checks and pins and sets the corresponding flags */

    // initialise variables
    bool maybe_pin = false;
    int num_checks = 0;             // number of checks
    int num_pinned = 0;             // number of pinned pieces
    int line_sq;                    // square along a line being checked
    int attack_line;                // line attack comes from
    int direction = 0;              // direction attack comes from
    int attack_sq;                  // square from which an attack comes
    int attack_piece;               // the piece attacking us
    int attack_mod;                 // attack modifier

    // loop through any attacks on the king
    int loop_num = king_pad_struct.attack_me.size() / 3;
    for (int i = 0; i < loop_num; i++) {

        // extract the details of the attack
        attack_sq = king_pad_struct.attack_me[(i * 3) + 0];
        attack_piece = king_pad_struct.attack_me[(i * 3) + 1];
        attack_mod = king_pad_struct.attack_me[(i * 3) + 2];

        // reset pin counter to zero
        num_pinned = 0;

        // is a piece attacking (checking) us
        if (attack_mod == -1) {
            num_checks += 1;
            check = true;
        }

        // if a piece is pinned to us -> nb in c++: -21/10 = -2, in python -21//10 = -3
        else if (attack_mod / 10 == -2) {
            maybe_pin = true;
        }
        else { // it is an attack blocked by a friendly piece
            continue;
        }

        // pins/blocking check is not possible if the attacker is a knight
        if (attack_piece == 2) {
            // however it is possible to capture the knight
            block_check.push_back(attack_sq);
            continue;
        }

        // what is the line of this check/pin
        attack_line = attack_sq - king_index;

        // determine the direction the attack comes from
        if (attack_line > 0) {
            if (attack_line % 9 == 0) direction = 9;
            else if (attack_line % 10 == 0) direction = 10;
            else if (attack_line % 11 == 0) direction = 11;
            else direction = 1;
        }
        else {
            if (attack_line % 9 == 0) direction = -9;
            else if (attack_line % 10 == 0) direction = -10;
            else if (attack_line % 11 == 0) direction = -11;
            else direction = -1;
        }

        // set this vector to empty
        std::vector<int> move_set;

        // now we know what direction the piece is pinned/checked along
        for (int l = 1; l < 8; l++) {

            line_sq = king_index + direction * l;

            // if the square is out of bounds, break
            if (board.arr[line_sq] == 7 or
                board.arr[line_sq] == -7) break;
            // if the line square is empty
            else if (board.arr[line_sq] == 0) {
                // then the square is along the pin/check line
                move_set.push_back(line_sq);
            }
            // if we reach the piece that is pinning/checking us
            else if (line_sq == attack_sq) {
                // save this square, then break
                move_set.push_back(line_sq);
                break;
            }
            // if we reach the piece that is pinned
            else if (maybe_pin and board.arr[line_sq] * player_colour > 0) {
                pinned_piece.push_back(line_sq);
                // testing-----------
                pinned_moves.push_back(0);
                pinned_moves.push_back(line_sq);
                //-------------------
                num_pinned += 1;
            }
        }

        // now we have all the squares along the line

        // if it is a checking line
        if (attack_mod == -1) {
            // add the move set to block check
            block_check.insert(block_check.end(), move_set.begin(),
                move_set.end());
        }
        // if it is a pin
        else if (maybe_pin) {
            // confirm we do in fact have a pinned piece
            if (num_pinned == 1) {
                pin = true;
                // add the move set to block check
                pinned_moves.insert(pinned_moves.end(), move_set.begin(),
                    move_set.end());
            }
            // if we found two pinned pieces, it is not a pin at all
            /* the cause of this error is not clear, but I believe it arises
            * from bishops/queens along the same diagonal as a king being then
            * pinned by a piece */
            else if (num_pinned > 1) {
                // remove the pinned pieces we thought we had
                while (num_pinned > 0) {
                    pinned_piece.pop_back();
                    // testing------------------
                    pinned_moves.pop_back();
                    pinned_moves.pop_back();
                    // -------------------------
                    num_pinned -= 1;
                }
            }
            else {
                // for testing: print details about the error
                // py_print_board(board, false);
                // py::print("Num pinned = ", num_pinned);
                // py::print("Move set = ", move_set);
                // py::print("Pinned piece = ", pinned_piece);
                // py::print("Piece colour = ", player_colour);
                throw std::runtime_error("We found the wrong number of pinned pieces");
            }
            // reset before next loop
            maybe_pin = false;
        }
    }

    return num_checks;
}

bool king_walks(Board& board, int king_index, int player_colour,
    std::vector<int>& walk_squares, piece_attack_defend_struct& king_pad_struct) {
    /* This function determines whether a king can walk to any nearby squres,
    returning true if so and updating the move vector */

    // initialise variables
    bool is_attacked;               // is a square attacked
    bool any_escape = false;        // are there any safe king walks
    int check_square;               // square being looked at for check

    // firstly, remove the king from his square (so he can't block any checks)
    board.set(king_index, 0);

    // loop through the squares the king can move to
    int loop_num = king_pad_struct.attack_list.size() / 3;
    for (int i = 0; i < loop_num; i++) {

        check_square = king_pad_struct.attack_list[(i * 3)];

        // is this square attacked by an opposing piece
        is_attacked = is_in_check(board, check_square, player_colour);

        // if the square is not attacked, the king can move there
        if (not is_attacked) {
            // we add a new element to walk_squares
            walk_squares.push_back(king_index);
            walk_squares.push_back(king_pad_struct.attack_list[(i * 3)]);
            // determine the move modifier
            if (board.arr[check_square] == 0) {
                walk_squares.push_back(1);          // move = 1
            }
            else {
                walk_squares.push_back(2);          // capture = 2
            }

            // the king is not trapped
            any_escape = true;
        }
    }

    // return the king to board
    board.set(king_index, 6 * player_colour);

    return any_escape;
}

bool is_in(std::vector<int> list, int target_value) {
    /* This function determines whether a target integer is contained within
    a list, returning true if so, and false otherwise */
    for (unsigned int i = 0; i < list.size(); i++) {
        if (target_value == list[i]) {
            return true;
        }
    }
    return false;
}

void get_my_pin_moves(std::vector<int> pinned_moves, std::vector<int>& my_moves,
    int target_piece_sq) {
    /* This function extracts the legal moves that can be made by a pinned piece
    from the vector pinned_moves, which has the structure:
        {0, 24, 25, 26, 35, 0, 33, 43, 53, 63}
    0 indicates a new pinned piece, which is then directly followed by the square
    the pinned piece is on. Following that are the legal moves the pinned piece
    can make, so those are what is extracted by this function */

    bool found_piece = false;       // have we found the pinned piece we want

    for (unsigned int i = 0; i < pinned_moves.size(); i++) {
        // have we reached a new pinned piece
        if (pinned_moves[i] == 0) {
            i += 1;
            // did we already find the piece we wanted
            if (found_piece) {
                break;
            }
            // if this piece is the one we are looking for
            if (pinned_moves[i] == target_piece_sq) {
                found_piece = true;
            }
        }
        // if we found the piece we need, and the target square comes up
        else if (found_piece) {
            // add this square as available to be moved to
            my_moves.push_back(pinned_moves[i]);
        }
    }

    // testing, check that the piece was indeed pinned
    if (not found_piece) {
        throw std::runtime_error("get_my_pinned_moves was given a non pinned piece");
        // return; // piece isn't pinned, can do what it likes
    }
}

bool is_checkmate(Board& board, int king_index, int player_colour,
    piece_attack_defend_struct& king_pad_struct) {
    /* This function checks if the king is in checkmate */

    bool check = false;
    bool pin = false;
    std::vector<int> block_check;
    std::vector<int> pinned_moves;
    std::vector<int> pinned_piece;
    std::vector<int> walk_squares;
    std::vector<int> castle_moves;
    std::vector<int> legal_moves;
    std::vector<int> my_pinned_moves;

    // determine if there are pins or checks (involving the king)
    int num_checks = find_pins_checks(board, check, pin, king_index, player_colour,
        block_check, pinned_moves, pinned_piece, king_pad_struct);

    /* FOR TESTING
    std::cout << "Number of checks is " << num_checks << '\n';
    std::cout << "Block check is [";
    for (int i = 0; i < block_check.size(); i++) {
        std::cout << " " << block_check[i];
    }
    std::cout << " ]\n";
    */

    // if there are no checks, it cannot be checkmate
    if (num_checks == 0) {
        return false;
    }

    // next, we find if the king can walk to any squares

    // determine if the king has any safe squares to walk to, add these moves to norm_moves
    bool safe_squares = king_walks(board, king_index, player_colour, legal_moves,
        king_pad_struct);

    // if there are any king walk squares, it cannot be checkmate
    if (safe_squares) {
        return false;
    }

    //// determine if there are any legal castling moves to be made by the king
    //bool any_castles = list_of_castles(board, player_colour, legal_moves);

    //// if there are any castling moves, it cannot be checkmate
    //if (any_castles) {
    //    return false;
    //}

    // now we have established that the king himself has no legal moves

    // if it is a double check, it must be checkmate since it cannot be blocked
    if (num_checks > 1) {
        return true;
    }

    // initialise some more variables
    int pawn_square_one;
    int pawn_square_two;
    int pawn_move;
    int pawn_start;
    int defend_sq;
    int defend_piece;
    int defend_mod;
    piece_attack_defend_struct pad_struct;

    if (player_colour == 1) {
        pawn_move = 10;
        pawn_start = 3;
    }
    else if (player_colour == -1) {
        pawn_move = -10;
        pawn_start = 8;
    }
    else throw std::invalid_argument("is_checkmate got player colour not equal to 1/-1");

    // last, we loop through block check to see if a piece can block
    for (int square : block_check) {

        // analyse the square as if a king were on it
        pad_struct = piece_attack_defend(board, square, 6, player_colour);

        // pawn moves are the only type not included in defend_me
        // a square must be empty for a pawn to move to it
        if (board.arr[square] == 0) {

            // what are the squares a pawn would have to move from
            pawn_square_one = square + pawn_move * -1;

            // could a pawn do a double move to get here
            if (square + pawn_move * 2 / 10 == pawn_start) {
                if (board.arr[square + pawn_move * -1] == 0) {
                    // yes, so save this square
                    pawn_square_two = square + pawn_move * -2;
                }
                // otherwise save an impossible square
                else pawn_square_two = 0;
            }
            else pawn_square_two = 0;

            // loop through all the pieces that see this square to find pawns
            for (int view : pad_struct.piece_view) {
                // if the piece seen is where a pawn can move from
                if (view == pawn_square_one or
                    view == pawn_square_two) {
                    // is one of our pawns
                    if (board.arr[view] == 1 * player_colour) {
                        // is this piece pinned
                        if (pin and is_in(pinned_piece, view)) {
                            my_pinned_moves.clear();   // reset this to empty
                            // find what my pinned moves are
                            get_my_pin_moves(pinned_moves, my_pinned_moves, view);
                            // if our square is not a possible pin move
                            if (not is_in(my_pinned_moves, square)) {
                                continue; // this move is not legal
                            }
                        }

                        // hence this pawn can block checkmate
                        return false;
                    }
                }
            }
        }

        // now check if other pieces can block checkmate
        int num_loops = pad_struct.defend_me.size() / 3;
        for (int i = 0; i < num_loops; i++) {

            // extract information about the defending piece
            defend_sq = pad_struct.defend_me[(i * 3) + 0];
            defend_piece = pad_struct.defend_me[(i * 3) + 1];
            defend_mod = pad_struct.defend_me[(i * 3) + 2];

            // check if the piece in question is a king - it can't block itself
            if (defend_piece == 6) {
                continue;
            }

            // check if the piece is a pawn
            if (defend_piece == 1) {
                // it cannot move to this square unless it is a capture
                if (board.arr[square] == 0) {
                    continue;
                }
            }

            // is this piece pinned
            if (pin and is_in(pinned_piece, defend_sq)) {
                my_pinned_moves.clear();   // reset this to empty
                // find what my pinned moves are
                get_my_pin_moves(pinned_moves, my_pinned_moves, defend_sq);
                // if our square is not a possible pin move
                if (not is_in(my_pinned_moves, square)) {
                    continue; // this move is not legal
                }
            }

            // hence we aren't pinned, the move is legal if it is a move
            if (defend_mod == 1) {
                // this move blocks check, so it isn't checkmate
                return false;
            }
        }
    }

    // if we get here, we have no legal moves and it is checkmate
    return true;
}

bool test_checkmate(Board& board, int player_colour) {
    /* This function tests is_checkmate function by automating finding the king */

    bool mate;
    int king_index;
    int index;
    piece_attack_defend_struct king_pad_struct;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {

            index = (2 + i) * 10 + (j + 1);

            // have we found the king
            if (board.arr[index] == player_colour * 6) {

                king_index = index;
                king_pad_struct = piece_attack_defend(board, king_index, 6, player_colour);

                mate = is_checkmate(board, king_index, player_colour, king_pad_struct);

                return mate;

            }
        }
    }

    throw std::runtime_error("test_checkmate could not find the king");
    return false;
}

int tempo_check(Board& board, bool white_to_play) {
    /* This function is used to see who is ahead in tempo */

    int tempo;

    if (white_to_play) tempo = 0;
    else tempo = -1;

    // white pieces
    //if (board.arr[21] != 4) tempo += 1;
    if (board.arr[22] != 2) tempo += 1;
    if (board.arr[23] != 3) tempo += 1;
    if (board.arr[24] != 6) tempo += 1;
    //if (board.arr[25] != 5) tempo += 1;
    if (board.arr[26] != 3) tempo += 1;
    if (board.arr[27] != 2) tempo += 1;
    //if (board.arr[28] != 4) tempo += 1;

    // white pawns
    //if (board.arr[31] != 1) tempo += 1;
    //if (board.arr[32] != 1) tempo += 1;
    //if (board.arr[33] != 1) tempo += 1;
    if (board.arr[34] != 1) tempo += 1;
    if (board.arr[35] != 1) tempo += 1;
    if (board.arr[36] != 1) tempo += 1;
    //if (board.arr[37] != 1) tempo += 1;
    //if (board.arr[38] != 1) tempo += 1;

    // black pieces
    //if (board.arr[91] != -4) tempo -= 1;
    if (board.arr[92] != -2) tempo -= 1;
    if (board.arr[93] != -3) tempo -= 1;
    if (board.arr[94] != -6) tempo -= 1;
    //if (board.arr[95] != -5) tempo -= 1;
    if (board.arr[96] != -3) tempo -= 1;
    if (board.arr[97] != -2) tempo -= 1;
    //if (board.arr[98] != -4) tempo -= 1;

    // black pawns
    //if (board.arr[81] != -1) tempo -= 1;
    //if (board.arr[82] != -1) tempo -= 1;
    //if (board.arr[83] != -1) tempo -= 1;
    if (board.arr[84] != -1) tempo -= 1;
    if (board.arr[85] != -1) tempo -= 1;
    if (board.arr[86] != -1) tempo -= 1;
    //if (board.arr[87] != -1) tempo -= 1;
    //if (board.arr[88] != -1) tempo -= 1;

    return tempo;
}

// need a rate pawn structure function, could count up total num of pieces too

int linear_insert(int value, int start_index, std::vector<int>& vec)
{  /* This function inserts a value into a vector in ascending order, starting
   the search from the start_index. Returns the index added at */
    
    // if the vector is empty, add the value in
    if (vec.size() == 0) {
        vec.push_back(value);
        return 0;
    }
    // if the start index is beyond the end of the vector, add it at the end
    else if (start_index > vec.size() - 1) {
        vec.push_back(value);
        return vec.size() - 1;
    }
    // else, lets traverse the vector looking for where we can insert
    else {
        for (int i = start_index; i < vec.size(); i++) {
            if (value < vec[i]) {
                vec.insert(vec.begin() + i, value);
                return i;
            }
        }
    }

    // if we get here, we didn't find anywhere to insert the value
    vec.push_back(value);
}

std::vector<int> order_attackers_defenders(std::vector<int>& pieces,
    std::vector<int>& indexes, std::vector<int>& master_list)
{  /* This function orders a list of attackers or defenders, putting them in the
      order they would attack so the least valuable pieces attack first eg a pawn
      before a queen. It needs the indexes and master list to correctly handle
      pieces that get uncovered during a sequence of attacks. */

    std::vector<int> ordered_list;  // final list of ordered attackers/defenders

    // traverse the vector of pieces
    int i = 0;
    int j;
    for (int value : pieces) {

        // insert into our order list in ascending order
        j = linear_insert(value, 0, ordered_list);
        i += 1;

        // check if this piece attacking uncovers another
        int uncover = (indexes[i - 1] + 1) * 3;
        if (master_list.size() > uncover and
            (master_list[uncover + 2] == -2 or
                master_list[uncover + 2] == 2)) {
            // we have another piece to add to the list
            j = linear_insert(master_list[uncover + 1], j, ordered_list);
            i += 1;
            // check if this piece attacking uncovers another
            uncover += 3;
            if (master_list.size() > uncover and
                (master_list[uncover + 2] == -3 or
                    master_list[uncover + 2] == 3)) {
                // we have another piece to add to the list
                j = linear_insert(master_list[uncover + 1], j, ordered_list);
                i += 1;
            }
        }

        // TESTING CODE: prevent vector overflow for line -> int uncover = (indexes[i - 1] + 1) * 3;
        if (i > pieces.size() - 1) {
            break;
        }
    }

    //// FOR TESTING
    //std::cout << "The ordered list is: ";
    //for (int x : ordered_list) {
    //    std::cout << x << ", ";
    //}
    //std::cout << '\n';

    return ordered_list;
}

phase_struct determine_phase(Board& board, bool white_to_play) {
    /* This function determines the phase of play on the board, and returns
    a structure containing all the details */

    phase_struct phase;
    
    /* the first step is to determine the phase:
    *       1 = early game, prior to castling
    *       2 = middle game
    *       3 = end game, no queens and few pieces
    */

    phase.phase = 1;

    // if both players have castled or lost rights, phase 2
    if (board.arr[0] == -7 and board.arr[1] == -7 and
        board.arr[2] == -7 and board.arr[3] == -7) {

        phase.phase = 2;

        // if enough pieces have left the board
        bool wQ_gone = true;
        bool bQ_gone = true;
        int piece_value = 0;
        int index;

        // count up the value of the pieces left on the board
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {

                index = (2 + i) * 10 + (j + 1);

                // add up the value of all the pieces
                if (board.arr[index] > 0) {
                    piece_value += board.arr[index];
                }
                else if (board.arr[index] < 0) {
                    piece_value -= board.arr[index];
                }

                // check if we have reached a queen
                if (board.arr[index] == 5) {
                    wQ_gone = false;
                }
                if (board.arr[index] == -5) {
                    bQ_gone = false;
                }
            }
        }

        // have enough pieces gone to make this phase 3?
        if (piece_value < 32) {
            if (wQ_gone and bQ_gone) {
                phase.phase = 3;
            }
        }
    }

    // testing
    //std::cout << "The phase is " << phase.phase << '\n';

    // now we have the phase of the game, fill in the relevant details
    if (phase.phase == 1) {
        phase.pawn_mod = 15;            // 25 means approx max boost of 0.25 points
        phase.knight_mod = 30;
        phase.bishop_mod = 30;
        phase.rook_mod = 10;
        phase.queen_mod = 25;
        phase.king_mod = 25;
        phase.king_vunerability_mod = 1000;
        
        phase.piece_defended_boost = 50;
        phase.pawn_pawn_defender_boost = 50;
        phase.passed_pawn_bonus = 1000;
        phase.tempo_bonus = 200;
        phase.castle_bonus = 300;

        phase.null_attack = 1000;                 // divide null evals by 20
        phase.null_defend = 1000;                 // divide null evals by 20
        phase.null_active = 1000;               // divide null evals by 1000
    }
    else if (phase.phase == 2) {
        phase.pawn_mod = 20;
        phase.knight_mod = 40;
        phase.bishop_mod = 40;
        phase.rook_mod = 40;
        phase.queen_mod = 40;
        phase.king_mod = 25;
        phase.king_vunerability_mod = 1500;

        phase.piece_defended_boost = 50;
        phase.pawn_pawn_defender_boost = 50;
        phase.passed_pawn_bonus = 1000;
        phase.tempo_bonus = 0;
        phase.castle_bonus = 300;

        phase.null_attack = 1000;                 // divide null evals by 20
        phase.null_defend = 1000;                 // divide null evals by 20
        phase.null_active = 1000;               // divide null evals by 1000
    }
    else if (phase.phase == 3) {
        phase.pawn_mod = 50;
        phase.knight_mod = 40;
        phase.bishop_mod = 50;
        phase.rook_mod = 75;
        phase.queen_mod = 75;
        phase.king_mod = 100;
        phase.king_vunerability_mod = 500;
        
        phase.piece_defended_boost = 50;
        phase.pawn_pawn_defender_boost = 100;
        phase.passed_pawn_bonus = 1000;
        phase.tempo_bonus = 0;
        phase.castle_bonus = 0;

        phase.null_attack = 1000;                 // divide null evals by 20
        phase.null_defend = 1000;                 // divide null evals by 20
        phase.null_active = 1000;               // divide null evals by 1000
    }

    // now adjust the evaluation depending on the board state
    phase.evaluation_adjust = 0;

    // reward castling by checking the two 'has castled' booleans
    if (board.arr[101] == 7) phase.evaluation_adjust += phase.castle_bonus;
    else {
        // if not castled, punish if castle rights have been lost
        if (board.arr[0] == -7 and board.arr[1] == -7)
            phase.evaluation_adjust -= phase.castle_bonus / 2;
    }
    if (board.arr[102] == 7) phase.evaluation_adjust -= phase.castle_bonus;
    else {
        // if not castled, punish if castle rights have been lost
        if (board.arr[2] == -7 and board.arr[3] == -7)
            phase.evaluation_adjust += phase.castle_bonus / 2;
    }

    // reward developing the pieces quickly
    if (phase.phase == 1) {
        // find out who is ahead in tempo
        int tempo = tempo_check(board, white_to_play);

        // give a favourable evalution to player ahead in tempo
        phase.evaluation_adjust += tempo * phase.tempo_bonus;
    }

    // reward good pawn structure
    /* currently this is rewarded when pawns defend pawns, might be fine to leave as it is */

    // reward passed pawns
    /* currently, passed pawns are rewarded in eval_piece, might be fine to leave as it is */

    return phase;
}

int is_passed_pawn(Board& board, int our_colour, int square_num) {
    /* This function determines if a pawn is passed. It returns 0 if the pawn
       is not passed, and if it is, it returns the number of ranks away from
       promotion. So, white passed pawn on 7th rank returns 1, 6th rank 2 etc */

    int col = square_num % 10;
    int row = square_num / 10;

    // loop downwards along the column we are on
    for (int i = 1; i < 8; i++) {
        // have we got to the end of the board without finding an opposition pawn
        if (board.arr[(row + i * our_colour) * 10 + col] == -7 or
            board.arr[(row + i * our_colour) * 10 + col] == 7) {
            return i - 1;
        }
        // does this column or either adjacent one contain an opposition pawn
        if (board.arr[(row + i * our_colour) * 10 + (col - 1)] == -1 * our_colour or
            board.arr[(row + i * our_colour) * 10 + (col - 0)] == -1 * our_colour or
            board.arr[(row + i * our_colour) * 10 + (col + 1)] == -1 * our_colour) {
            return 0;
        }
    }
}

int piece_value(int piece_num) {
    /* This function converts a piece number into a piece value */

    if (piece_num == 1 or piece_num == -1)      return 1;
    else if (piece_num == 2 or piece_num == -2) return 3;
    else if (piece_num == 3 or piece_num == -3) return 3;
    else if (piece_num == 4 or piece_num == -4) return 5;
    else if (piece_num == 5 or piece_num == -5) return 9;
    else if (piece_num == 6 or piece_num == -6) return 4;
    else {
        throw std::invalid_argument{ "bad input to piece_value" };
    }
}

int eval_piece(Board& board, bool white_to_play, int square_num,
    phase_struct& phase, int our_piece, int our_colour,
    bool& mate, piece_attack_defend_struct& pad_struct)
{   
    /* This function determines the value of a piece */

    int evaluation = 0;         // piece evaluation
    int value;                  // value modifier of the piece
    int potential;              // max possible number of attacked squares
    int attack_score = 0;       // rating for the attack power of the piece
    int attack_bonus;           // evaluation bonus for attack power

    bool pawn = false;          // are we a pawn
    bool passed_pawn = false;   // are we a passed pawn
    mate = false;               // set to default

    int target_rows[4];                     // which rows should this piece target
    int centre_cols[4] = { 3, 4, 5, 6 };    // central board columns

    // which side does this piece attack
    if (our_colour == 1) {
        target_rows[0] = 9;
        target_rows[1] = 8;
        target_rows[2] = 7;
        target_rows[3] = 6;
    }
    else if (our_colour == -1) {
        target_rows[0] = 2;
        target_rows[1] = 3;
        target_rows[2] = 4;
        target_rows[3] = 5;
    }
    else {
        throw std::invalid_argument("eval piece recieved our_colour != -1 or 1");
    }

    // are we a pawn.....? are we a passed pawn.....?

    // go through the squares this piece attacks, and rate its attack strength
    int loop_num = pad_struct.attack_list.size() / 3;
    for (int i = 0; i < loop_num; i++) {

        // extract the details of the attack
        int attack_sq = pad_struct.attack_list[(i * 3) + 0];
        // int attack_piece = pad_struct.attack_list[(i * 3) + 1];
        int attack_mod = pad_struct.attack_list[(i * 3) + 2];

        // bonus applied upon attacking a square
        constexpr int bump = 5;

        attack_score += bump;

        // score the attack better if it lands in the opposition half of the board
        if (attack_sq / 10 == target_rows[0] or
            attack_sq / 10 == target_rows[1] or
            attack_sq / 10 == target_rows[2] or
            attack_sq / 10 == target_rows[3]) {

            attack_score += bump;

            // score it better if it strikes in the centre
            if (attack_sq % 10 == centre_cols[0] or
                attack_sq % 10 == centre_cols[1] or
                attack_sq % 10 == centre_cols[2] or
                attack_sq % 10 == centre_cols[3]) {

                attack_score += bump;
            }
        }

        

        /*
        // score the attack better if it lands in the opposition half of the board
        if (attack_sq / 10 == target_rows[0] or
            attack_sq / 10 == target_rows[1] or
            attack_sq / 10 == target_rows[2] or
            attack_sq / 10 == target_rows[3]) {
            // score it better if it strikes in the centre
            if (attack_sq % 10 == centre_cols[0] or
                attack_sq % 10 == centre_cols[1] or
                attack_sq % 10 == centre_cols[2] or
                attack_sq % 10 == centre_cols[3]) {

                attack_score += 15;  // attack in their half, in the centre
            }
            else {
                attack_score += 10;  // attack in their half
            }
        }
        else {
            attack_score += 5;      // basic attack
        }
        */

        // no additional bonus if it attacks an empty square
        if (attack_mod == 0) {
            // pass
        }
        // score the attack better if it attacks a piece
        else if (attack_mod == -1 or
            attack_mod == -2 or
            attack_mod == -3) {

            attack_score += 10;      // attack on a piece
        }
        // score the attack better if it is a pin or discovery
        else { // attack mod is -20 to -36

            /* It would be great to add in better logic here, to reward good pins
               better and rate bad pins (eg attack through a pawn) lower */

            attack_score += 5;      // attack through a piece (pin/discovery)
        }
    }

    // convert attack score to a piece bonus, weighted by max number of attacked sqs
    // lookup the piece value and potential
    /* Potential is a score based on the maximum number of squares a piece can
       attack. One point given for squares in an opponents half, 0.5 points given
       for squares in our half. */
    if (our_piece == 1) {       // pawn   2     (0)
        pawn = true;
        value = 1;
        potential = 2;
        attack_bonus = phase.pawn_mod * (attack_score / potential);
    }
    else if (our_piece == 2) {  // knight 6     (2)
        value = 3;
        potential = 7;
        attack_bonus = phase.knight_mod * (attack_score / potential);
    }
    else if (our_piece == 3) {  // bishop 6     (7)
        value = 3;
        potential = 10;
        attack_bonus = phase.bishop_mod * (attack_score / potential);
    }
    else if (our_piece == 4) {  // rook   10    (4)
        value = 5;
        potential = 12;
        attack_bonus = phase.rook_mod * (attack_score / potential);
    }
    else if (our_piece == 5) {  // queen  16    (11)
        value = 9;
        potential = 22;
        attack_bonus = phase.queen_mod * (attack_score / potential);
    }
    else if (our_piece == 6) {  // king   8     (0)
        value = 4;
        potential = 8;
        attack_bonus = phase.king_mod * (attack_score / potential);
    }
    else {
        std::cout << "our_piece was " << our_piece << '\n';
        print_board(board);
        throw std::invalid_argument("eval_piece has recieved our_piece != 1,2,3,4,5,or 6");
    }

    // if we are a passed pawn, give a further bonus based on our square
    if (pawn) {
        int pp = is_passed_pawn(board, our_colour, square_num);
        if (pp != 0) {
            // give a better boost if the pawn is closer to promotion
            attack_bonus += phase.passed_pawn_bonus
                - ((pp - 1) * (phase.passed_pawn_bonus / 7));
        }
    }

    /* max attack_score / potential = 10
       pawn value = 1000
       so max attack bonus = (eg) 50 * 180/22 = 400
       400 = 0.4 pawns */

       // apply the value of the piece (eg knight worth 3pts = 3000)
    evaluation += value * 1000;

    // apply the attack bonus
    evaluation += attack_bonus;

    //std::cout << "The piece on square " << square_num << " with value " << value
    //    << " has attack_bonus of " << attack_bonus;

    // now determine the defensive strength of the piece
    int net_trade = 0;              // net outcome of trades
    int defend_piece = value;       // value of piece on the square
    bool king_involved = false;     // is the king involved in tactics

    // if the piece in question is the king, things are different
    if (our_piece == 6) {

        int checks = 0;
        int vunerability_score = 0; // rating for the defensive vunerability of the king
        int vunerability_penalty;   // evaluation penalty for this defensive vunerability

        // loop through all the squares the king is vunerable from
        int loop_num = pad_struct.attack_me.size() / 3;
        for (int i = 0; i < loop_num; i++) {

            // extract the details of the attack
            int attack_sq = pad_struct.attack_me[(i * 3) + 0];
            // int attack_piece = pad_struct.attack_me[(i * 3) + 1];
            int attack_mod = pad_struct.attack_me[(i * 3) + 2];

            // if the square is empty
            if (attack_mod == 0) {
                vunerability_score += 1;
            }
            // if it contains an attacking piece
            else if (attack_mod == -1) {
                vunerability_score += 10;
                checks += 1;
            }

            // more vunerable if attacked from the opposition half of the board
            if (attack_sq / 10 == target_rows[0] or
                attack_sq / 10 == target_rows[1] or
                attack_sq / 10 == target_rows[2] or
                attack_sq / 10 == target_rows[3]) {

                vunerability_score += 1;
            }
            // more vunerable if attacked from the centre
            if (attack_sq % 10 == centre_cols[0] or
                attack_sq % 10 == centre_cols[1] or
                attack_sq % 10 == centre_cols[2] or
                attack_sq % 10 == centre_cols[3]) {

                vunerability_score += 1;
            }
        }

        // check if the king is in checkmate
        if (checks != 0) {
            mate = is_checkmate(board, square_num, our_colour, pad_struct);
            // if it is checkmate, return 0, having changed mate& to true
            if (mate) {
                //// FOR TESTING
                //std::cout << "eval piece found a mate, here is board:\n";
                //print_board(board, false);
                return 0;
            }
        }

        // compute the vunerability of the king
        vunerability_penalty = (phase.king_vunerability_mod *
            vunerability_score) / 35; // 35 = max no. of squares attacked from

        // apply this vunerability penalty
        evaluation -= vunerability_penalty;

        //// FOR TESTING
        //std::cout << ". King vunerability penalty is " << vunerability_penalty << '\n';
    }
    // else if the piece is not a king
    else {
        //else if (pad_struct.attack_me.size() != 0) {

        int defend_bonus = 0;       // boost for being defended by our pieces

        std::vector<int> attacker_pieces;   // vector of attacking pieces
        std::vector<int> attacker_indexes;  // vector of indexes for above vector
        std::vector<int> defender_pieces;   // vector of defending pieces
        std::vector<int> defender_indexes;  // vector of indexes for above vector

        // gather any attackers
        int loop_num = pad_struct.attack_me.size() / 3;
        for (int i = 0; i < loop_num; i++) {

            // extract the details of the attack
            // int attack_sq = pad_struct.attack_me[(i * 3) + 0];
            int attack_piece = pad_struct.attack_me[(i * 3) + 1];
            int attack_mod = pad_struct.attack_me[(i * 3) + 2];

            if (attack_mod == -1) {
                attacker_pieces.push_back(attack_piece);
                attacker_indexes.push_back(i);
            }
        }

        // gather any defenders
        loop_num = pad_struct.defend_me.size() / 3;
        for (int i = 0; i < loop_num; i++) {

            // extract the details of the defend
            // int defend_sq = pad_struct.defend_me[(i * 3) + 0];
            int defend_piece = pad_struct.defend_me[(i * 3) + 1];
            int defend_mod = pad_struct.defend_me[(i * 3) + 2];

            if (defend_mod == 1) {
                defender_pieces.push_back(defend_piece);
                defender_indexes.push_back(i);
            }
        }
        //std::cout << "--------------------\n";

        //std::cout << "attacker pieces is: ";
        //for (int i : attacker_pieces) {
        //    std::cout << i << ", ";
        //}
        //std::cout << "\n";

        //std::cout << "attack_indexes is: ";
        //for (int i : attacker_indexes) {
        //    std::cout << i << ", ";
        //}
        //std::cout << "\n";

        //std::cout << "attacker me is: ";
        //for (int i : pad_struct.attack_me) {
        //    std::cout << i << ", ";
        //}
        //std::cout << "\n";

        // sort the attackers into the order they would attack (lowest value first)
        std::vector<int> attacker_order = order_attackers_defenders(attacker_pieces,
            attacker_indexes, pad_struct.attack_me);

        //std::cout << "attacker order is: ";
        //for (int i : attacker_order) {
        //    std::cout << i << ", ";
        //}
        //std::cout << "\n";

        // sort the defenders into the order they would defend (lowest value first)
        std::vector<int> defender_order = order_attackers_defenders(defender_pieces,
            defender_indexes, pad_struct.defend_me);

        //std::cout << "defender order is: ";
        //for (int i : defender_order) {
        //    std::cout << i << ", ";
        //}
        //std::cout << "\n";

        // is this piece defended
        if (defender_order.size() >= attacker_order.size()) {
            //// OLD CODE SCALED BOOST WITH NUMBER OF DEFENDERS
            //// we will recieve a boost for every extra defender we have
            //defend_bonus = phase.piece_defended_boost *
            //    (defender_order.size() - attacker_order.size());
            // NEW CODE GIVE ONLY THE BOOST FOR BEING DEFENDED
            // give us a boost if we are defended
            defend_bonus = phase.piece_defended_boost;
        }

        // are we a pawn protected by another pawn?
        if (pawn) {
            if (defender_order.size() > 0 and
                defender_order[0] == 1) {
                defend_bonus += phase.pawn_pawn_defender_boost;
                // are we a protected past pawn?
                if (passed_pawn) {
                    // reapply the defender boost
                    defend_bonus += phase.pawn_pawn_defender_boost;
                }
            }
        }

        // now apply any defensive bonus to the evalution
        evaluation += defend_bonus;

        //// TESTING
        //std::cout << ", the defend bonus is " << defend_bonus;

        // if we are attacked, we should check the outcome of trades
        if (attacker_order.size() > 0) {

            int trade_outcome = 0;          // trade outcome from immediate attack
            int square_value = value;       // value of our piece on its square

            // loop through every attacker, in the order that they would attack
            for (int i = 0; i < attacker_indexes.size(); i++) {

                // if there are no defenders left to take back
                if (defender_order.size() - i < 1) {
                    // the piece is lost
                    trade_outcome -= 1000 * square_value;
                    break;
                }

                // if the last defender is a king and capturing back is illegal
                if (defender_order[i] == 6 and attacker_order.size() > i + 1) {
                    // the piece is lost
                    trade_outcome -= 1000 * square_value;
                    break;
                }

                // if the next attacker is worth more than the piece it will capture
                if (piece_value(attacker_order[i]) > square_value) {
                    // taking is not worth it for us
                    break;
                }

                // if their attacker is their king, and our defender is not our king
                if (attacker_order[i] == 6 and defender_order[i] != 6) {
                    // it is illegal to capture
                    break;
                }

                // otherwise, they capture our piece on the square
                trade_outcome -= 1000 * square_value;

                // and we capture back
                trade_outcome += 1000 * piece_value(attacker_order[i]);

                // now we have a new piece on the square
                square_value = piece_value(defender_order[i]);

            }

            // check if trading is worse than simply losing the piece
            if (trade_outcome < -1000 * value) {
                // in this case we just give up the piece
                trade_outcome = -1000 * value;
            }

            //// TESTING
            //std::cout << " and the trade outcome is " << trade_outcome;

            // is it the attackers turn to move
            if ((white_to_play and our_colour == -1) or
                (not white_to_play and our_colour == 1)) {
                // this trade directly affects piece evaluation
                evaluation += trade_outcome;

                //// FOR TESTING
                //if (trade_outcome != 0) {
                //    std::cout << "The piece on square " << square_num
                //        << " has trade outcome " << trade_outcome << '\n';
                //}
            }
            // else it is our turn to move
            else {

                // this trade indirectly affects piece evaluation
                evaluation += trade_outcome / phase.null_active;

                //// FOR TESTING
                //std::cout << "The square " << square_num << " has a null trade outcome of "
                //    << trade_outcome << " / " << phase.null_active << " = "
                //    << trade_outcome / phase.null_active << '\n';
            }
        }
    }

    /* For the body of this function, + meant good evaluation
       now we adjust based on the piece colours:
       -> for a white piece, + means good evaluation (eg +3.4)
       -> for a black piece, - means good evaluation (eg -1.3) */

       //std::cout << '\n';

    return evaluation * our_colour;
}

int eval_board(Board& board, bool white_to_play) 
{  /* This function evalutes the board to decide who is winning */

    int evaluation = 0;         // the evaulation starts at zero
    int index;                  // board square index
    int piece_type;             // the type of piece
    int piece_colour;           // the colour of the piece
    bool mate = false;          // is it checkmate

    // determine the phase of the game
    phase_struct phase = determine_phase(board, white_to_play);

    // update the evaluation based on the phase of play
    evaluation += phase.evaluation_adjust;

    // loop through every square of the board
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {

            index = (2 + i) * 10 + (j + 1);

            // if the square contains a piece, find its evalution
            if (board.arr[index] != 0) {

                // extract the relevant information about the square
                if (board.arr[index] < 0) {
                    piece_type = board.arr[index] * -1;
                    piece_colour = -1;
                }
                else {
                    piece_type = board.arr[index];
                    piece_colour = 1;
                }

                // analyse the piece
                piece_attack_defend_struct pad_struct = piece_attack_defend(board,
                                                   index, piece_type, piece_colour);

                // evaluate the piece
                int piece_eval = eval_piece(board, white_to_play, index,
                    phase, piece_type, piece_colour, mate, pad_struct);

                // have we found checkmate
                if (mate) {
                    if (white_to_play) {
                        return WHITE_MATED;
                    }
                    else {
                        return BLACK_MATED;
                    }
                }

                //// FOR TESTING
                //std::cout << "The piece on square " << index
                //    << " has evaluation " << piece_eval.eval * piece_colour
                //    << " and null eval " << piece_eval.null_eval << '\n';

                // update the evalutation
                evaluation += piece_eval;
            }
        }
    }
   
    return evaluation;
}

std::vector<int> find_view(Board& board, int square_num)
{
    /* This function finds all the pieces in view of a square. Does not break at
    blockers, only at the edges of the board */

    // four pairs of pieces that account for all possible movements
    int move_group_one[4] = { 6, 2, 3, 4 };
    int move_group_two[4] = { 1, 2, 5, 5 };

    // initialise structures
    piece_moves_struct movements;               // store how pieces move

    // initialise variables
    int dest_sq = 0;            // destination square
    std::vector<int> view;      // output vector containing pieces in view

    // for each of the THREE pieces that cover all possible movements (ignore king/pawn)
    for (int i = 1; i < 4; i++) {

        // find out how this piece moves
        movements = how_piece_moves(move_group_one[i], 1);

        // check all the directions in which these pieces can move
        for (int move_dist : movements.directions) {

            // arithmetic begins at starting square
            dest_sq = square_num;

            // check all the squares these pieces can reach
            for (int j = 0; j < movements.depth; j++) {

                // check the next square along this line
                dest_sq += move_dist;

                // is this square on the board? If not, this line is over
                if (board.arr[dest_sq] == 7 or board.arr[dest_sq] == -7) break;

                // does this square contain a piece
                if (board.arr[dest_sq] != 0) {
                    view.push_back(dest_sq);
                }
            }
        }
    }

    return view;
}

total_legal_moves_struct total_legal_moves(Board& board, bool white_to_play) {
    /* This function generates all the possible legal moves in a position for
    one player. It also evaluates the board and every piece as if on the
    following move (not white_to_play), for use in the generate_moves function */

    // initialise structures
    total_legal_moves_struct data_array;
    piece_attack_defend_struct pad_struct;

    // initialise vectors
    std::vector<int> our_indexes;       // indexes of our pieces
    std::vector<int> their_indexes;     // indexes of their pieces
    std::vector<int> priority_moves;    // high priority moves eg capture,promote
    std::vector<int> norm_moves;        // normal moves, eg move without capture

    // initialise variables
    int evaluation = 0;                 // board evaluation
    int player_colour;                  // +1 for white, -1 for black
    int pawn_move;                      // distance a pawn moves
    int pawn_start;                     // starting rank for pawns
    int pawn_promote;                   // promotion rank for pawns
    int index;                          // board array index for a square
    int piece_type;                     // numerical value of a piece
    int piece_colour;                   // +1 for white, -1 for black
    int king_index = 0;                 // index for our king
    int d_ind;                          // board square index for data array
    int attack_sq;                      // square under attack
    int attacked_piece;                 // piece under attack
    int attack_mod;                     // attack modifier
    int move_mod;                       // move modifier
    bool i_am_pinned;                   // is a piece is pinned
    bool mate = false;                  // is it checkmate

    // set variables based on whose turn it is
    if (white_to_play) {
        player_colour = 1;
        pawn_move = 10;
        pawn_start = 3;
        pawn_promote = 9;
    }
    else {
        player_colour = -1;
        pawn_move = -10;
        pawn_start = 8;
        pawn_promote = 2;
    }

    // determine the phase on the board, save data
    phase_struct phase = determine_phase(board, white_to_play);
    evaluation += phase.evaluation_adjust;
    data_array.phase = phase.phase;
    data_array.phase_adjust = phase.evaluation_adjust;

    // set how sensitive we are to our opponents threats
    phase.null_active = phase.null_defend;

    // loop through every square on the board
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {

            index = (2 + i) * 10 + (j + 1);

            // does the square contain a piece
            if (board.arr[index] != 0) {

                // extract information about the piece
                if (board.arr[index] < 0) {
                    piece_type = board.arr[index] * -1;
                    piece_colour = -1;
                }
                else {
                    piece_type = board.arr[index];
                    piece_colour = 1;
                }

                // analyse this piece
                pad_struct = piece_attack_defend(board, index, piece_type, piece_colour);

                // evaluate the piece as if it were next move
                pad_struct.evaluation = eval_piece(board, not white_to_play, index,
                    phase, piece_type, piece_colour, mate, pad_struct);

                // save this piece data
                data_array.set_from_index(index, pad_struct);

                // continue to calculate the overall board evaluation
                evaluation += pad_struct.evaluation;

                /* checkmate doesn't matter if we are we are using eval_piece on the
                next move, it will get caught later if there is checkmate */
                //// did we find a checkmate
                //if (mate) {
                //    data_array.outcome = 1;
                //    if (white_to_play) {
                //        data_array.evaluation = WHITE_MATED;
                //    }
                //    else {
                //        data_array.evaluation = BLACK_MATED;
                //    }
                //    return data_array;
                //}

                // check if this piece is our king
                if (piece_type == 6 and piece_colour == player_colour) {
                    king_index = index;
                }
                // if not, save the location of this piece
                else if (piece_colour == player_colour) {
                    our_indexes.push_back(index);
                }
                else {
                    their_indexes.push_back(index);
                }
            }
        }
    }

    // now we have finished filling the data array

    // first, we look at the king - are any checks or pins on the board

    // make some variables to be set by find_pins_checks, king_walks, castling functions
    bool check = false;
    bool pin = false;
    std::vector<int> block_check;
    std::vector<int> pinned_moves;
    std::vector<int> pinned_piece;
    std::vector<int> walk_squares;
    std::vector<int> castle_moves;
    std::vector<int> my_pinned_moves; // this is used later

    if (king_index == 0) {
        std::cout << "King index not found for this board.\n";
        std::cout << "white_to_play was: " << white_to_play << '\n';
        print_board(board, false);
        throw std::runtime_error("king index not found for this board!");
    }

    // extract the analysis of the king
    piece_attack_defend_struct king_pad_struct = data_array.get_from_index(king_index);

    // determine if there are pins or checks (involving the king)
    int num_checks = find_pins_checks(board, check, pin, king_index, player_colour,
        block_check, pinned_moves, pinned_piece, king_pad_struct);

    // next, we find all of the kings legal moves

    // determine if the king has any safe squares to walk to, add these moves to norm_moves
    bool safe_squares = king_walks(board, king_index, player_colour, data_array.legal_moves,
        king_pad_struct);
    // determine if there are any legal castling moves to be made by the king
    bool any_castles = list_of_castles(board, player_colour, data_array.legal_moves);

    // now we have finished adding all the king moves

    // TESTING PRINTS
    /*
    std::cout << "Booleans: " << '\n';
    std::cout << "check is " << check << '\n';
    std::cout << "pin is " << pin << '\n';
    std::cout << "any castles is: " << any_castles << '\n';
    std::cout << "Pinned piece is: ";
    for (int g : pinned_piece) {
        std::cout << g << ", ";
    }
    std::cout << '\n';
    std::cout << "Block check is: ";
    for (int g : block_check) {
        std::cout << g << ", ";
    }
    std::cout << '\n';
    std::cout << "Pinned moves is: ";
    for (int g : pinned_moves) {
        std::cout << g << ", ";
    }
    std::cout << '\n';
    */

    // if it is a double check, no other pieces can legally move
    if (num_checks > 1) {
        // do nothing
        //return data_array;
    }
    else {
        // go through the rest of our pieces to find their moves
        for (int ind : our_indexes) {

            // find the corresponding data_array index
            d_ind = (ind - 21) - 2 * ((ind / 10) - 2);

            // if the piece is a pawn, then it has different legal moves
            if (board.arr[ind] == 1 or board.arr[ind] == -1) {

                // first, determine if this piece is pinned
                i_am_pinned = false;
                // if there is a pin on the board
                if (pin and is_in(pinned_piece, ind)) {
                    i_am_pinned = true;
                    my_pinned_moves.clear();   // reset this to empty
                    // find what my pinned moves are
                    get_my_pin_moves(pinned_moves, my_pinned_moves, ind);
                }

                // go through the attacks this pawn can make
                int num_attacks = data_array.square_info[d_ind].attack_list.size() / 3;
                for (int a = 0; a < num_attacks; a++) {
                    // extract the details of the attack
                    attack_sq = data_array.square_info[d_ind].attack_list[(a * 3)];
                    attacked_piece = data_array.square_info[d_ind].attack_list[(a * 3) + 1];
                    attack_mod = data_array.square_info[d_ind].attack_list[(a * 3) + 2];

                    // if we are pinned
                    if (i_am_pinned) {
                        // if the move does not maintain the pin, it isn't legal
                        if (not (is_in(my_pinned_moves, attack_sq))) {
                            continue;
                        }
                    }
                    // if our king is in check
                    if (check) {
                        // only a block or capture of checking piece is legal
                        if (not (is_in(block_check, attack_sq))) {
                            continue;
                        }
                    }
                    // hence, any piece we attack allows us to move to that square
                    if (attack_mod == -1) {
                        // if the destination square is empty, it is en passant
                        if (board.arr[attack_sq] == 0) {
                            move_mod = 3;
                            data_array.legal_moves.push_back(ind);
                            data_array.legal_moves.push_back(attack_sq);
                            data_array.legal_moves.push_back(move_mod);
                        }
                        // do we promote with this capture
                        else if (attack_sq / 10 == pawn_promote) {
                            // move_mod = 6: promote to knight
                            data_array.legal_moves.push_back(ind);
                            data_array.legal_moves.push_back(attack_sq);
                            data_array.legal_moves.push_back(6);
                            // move_mod = 7: promote to queen
                            data_array.legal_moves.push_back(ind);
                            data_array.legal_moves.push_back(attack_sq);
                            data_array.legal_moves.push_back(7);
                            // move_mod = 8: promote to bishop
                            data_array.legal_moves.push_back(ind);
                            data_array.legal_moves.push_back(attack_sq);
                            data_array.legal_moves.push_back(8);
                            // move_mod = 9: promote to rook
                            data_array.legal_moves.push_back(ind);
                            data_array.legal_moves.push_back(attack_sq);
                            data_array.legal_moves.push_back(9);
                        }
                        // else it is a completely normal capture
                        else {
                            move_mod = 2;
                            data_array.legal_moves.push_back(ind);
                            data_array.legal_moves.push_back(attack_sq);
                            data_array.legal_moves.push_back(move_mod);
                        }
                    }
                }

                // now check if this pawn can move forwards normally

                // if the square ahead of us is empty, we can move there
                if (board.arr[ind + pawn_move] == 0 and
                    (not check or is_in(block_check, ind + pawn_move)) and
                    not (i_am_pinned and
                        not (is_in(pinned_moves, ind + pawn_move)))) {
                    // is this move a promotion
                    if ((ind + pawn_move) / 10 == pawn_promote) {
                        // move_mod = 6: promote to knight
                        data_array.legal_moves.push_back(ind);
                        data_array.legal_moves.push_back(ind + pawn_move);
                        data_array.legal_moves.push_back(6);
                        // move_mod = 7: promote to queen
                        data_array.legal_moves.push_back(ind);
                        data_array.legal_moves.push_back(ind + pawn_move);
                        data_array.legal_moves.push_back(7);
                        // move_mod = 8: promote to bishop
                        data_array.legal_moves.push_back(ind);
                        data_array.legal_moves.push_back(ind + pawn_move);
                        data_array.legal_moves.push_back(8);
                        // move_mod = 9: promote to rook
                        data_array.legal_moves.push_back(ind);
                        data_array.legal_moves.push_back(ind + pawn_move);
                        data_array.legal_moves.push_back(9);
                    }
                    // else it is not a promotion
                    else {
                        move_mod = 1;
                        data_array.legal_moves.push_back(ind);
                        data_array.legal_moves.push_back(ind + pawn_move);
                        data_array.legal_moves.push_back(move_mod);
                    }
                }
                // if we haven't moved yet, we may be able to move two spaces
                if (ind / 10 == pawn_start and
                    board.arr[ind + pawn_move] == 0 and
                    board.arr[ind + 2 * pawn_move] == 0 and
                    (not check or is_in(block_check, ind + 2 * pawn_move)) and
                    not (i_am_pinned and
                        not (is_in(pinned_moves, ind + 2 * pawn_move)))) {
                    move_mod = 1;
                    data_array.legal_moves.push_back(ind);
                    data_array.legal_moves.push_back(ind + 2 * pawn_move);
                    data_array.legal_moves.push_back(move_mod);
                }
            }
            // else it is not a pawn or a king
            else {

                // first, determine if this piece is pinned
                i_am_pinned = false;
                // if there is a pin on the board
                if (pin and is_in(pinned_piece, ind)) {
                    i_am_pinned = true;
                    my_pinned_moves.clear();   // reset this to empty
                    // find what my pinned moves are
                    get_my_pin_moves(pinned_moves, my_pinned_moves, ind);
                }

                // go through the attacks this piece can make
                int num_attacks = data_array.square_info[d_ind].attack_list.size() / 3;
                for (int a = 0; a < num_attacks; a++) {
                    // extract the details of the attack
                    attack_sq = data_array.square_info[d_ind].attack_list[(a * 3)];
                    attacked_piece = data_array.square_info[d_ind].attack_list[(a * 3) + 1];
                    attack_mod = data_array.square_info[d_ind].attack_list[(a * 3) + 2];

                    // if we are pinned
                    if (i_am_pinned) {
                        // if the move does not maintain the pin, it isn't legal
                        if (not (is_in(my_pinned_moves, attack_sq))) {
                            continue;
                        }
                    }
                    // if our king is in check
                    if (check) {
                        // only a block or capture of checking piece is legal
                        if (not (is_in(block_check, attack_sq))) {
                            continue;
                        }
                    }

                    // if a square is attacked, we can move there
                    if (attack_mod == 0) {
                        move_mod = 1;
                        data_array.legal_moves.push_back(ind);
                        data_array.legal_moves.push_back(attack_sq);
                        data_array.legal_moves.push_back(move_mod);
                    }
                    // if a piece is attacked, we can capture
                    else if (attack_mod == -1) {
                        move_mod = 2;
                        data_array.legal_moves.push_back(ind);
                        data_array.legal_moves.push_back(attack_sq);
                        data_array.legal_moves.push_back(move_mod);
                    }
                }
            }
        }
    }

    // now we have finished going through every piece and all legal moves

    // we can now determine the outcome of the position

    data_array.evaluation = evaluation;

    // if there are no legal moves
    if (data_array.legal_moves.size() == 0) {
        // if we are in check, it must be checkmate
        if (check) {
            // we should never get here!
            // checkmate is detected earlier in this function
            data_array.outcome = 1;
            if (white_to_play) {
                data_array.evaluation = WHITE_MATED;
            }
            else {
                data_array.evaluation = BLACK_MATED;
            }
        }
        // if we aren't in check, it must be a draw
        else {
            data_array.outcome = 2;
            data_array.evaluation = 0;
        }
    }
    // else there are legal moves, the game is not over
    else {
        data_array.outcome = 0;
    }

    return data_array;
}

void ordered_insert_moves(move_struct& new_move, std::vector<move_struct>& vec,
    bool white_to_play) {
    /* This function inserts a move into a vector of moves, maintaining an
    order of best to worst (which depends on whether it is white or black's turn) */

    int sign;

    if (white_to_play) {
        sign = 1;
    }
    else {
        sign = -1;
    }

    // if the vector is empty, simply put the new move in
    if (vec.size() == 0) {
        vec.push_back(new_move);
        return;
    }

    for (int i = 0; i < vec.size(); i++) {
        // if the new evaluation better than this one
        if (new_move.evaluation * sign > vec[i].evaluation * sign) {
            vec.insert(vec.begin() + i, new_move);
            return;
        }
    }

    // if we got to the end of the vector, insert last
    vec.push_back(new_move);
}

bool verify_checkmate(Board& board, bool white_to_play)
{
    /* This is a testing function that verifies that the is_checkmate function
    is working properly */

    bool mate = false;
    bool tlm_mate = false;
    int king_index;
    int index;
    int player_colour;
    piece_attack_defend_struct king_pad_struct;

    if (white_to_play) player_colour = 1;
    else player_colour = -1;

    // find out from total_legal_moves if it is checkmate
    total_legal_moves_struct tlm_struct = total_legal_moves(board, white_to_play);

    if (tlm_struct.outcome == 1) tlm_mate = true;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {

            index = (2 + i) * 10 + (j + 1);

            // have we found the king
            if (board.arr[index] == player_colour * 6) {

                king_index = index;
                king_pad_struct = piece_attack_defend(board, king_index, 6, player_colour);

                mate = is_checkmate(board, king_index, player_colour, king_pad_struct);

                if (mate == tlm_mate) {
                    std::cout << "Verify checkmate, both agree mate is: " << mate << '\n';
                    return true;
                }
                else {
                    std::cout << "Verify checkmate, disagreement. tlm says "
                        << tlm_mate << ", is_checkmate says " << mate << '\n';
                    std::cout << "The board is:\n";
                    print_board(board, false);
                    return false;
                }
            }
        }
    }

    throw std::runtime_error("test_checkmate could not find the king");
    return false;
}

generated_moves_struct generate_moves(Board& board, bool white_to_play) {
    /* This function finds the best moves to play in a given board */

    generated_moves_struct generated_moves;
    piece_attack_defend_struct temp_pad_struct;     // for piece analysis
    std::vector<int> piece_view;                    // piece views
    bool mate = false;                              // checkmate found?
    bool next_to_play = not white_to_play;          // who plays next

    // save the starting board
    generated_moves.base_board = board;

    // first, get a list of all the total legal moves in the position
    total_legal_moves_struct tlm_struct = total_legal_moves(board, white_to_play);
    //generated_moves.base_evaluation = tlm_struct.evaluation;
    generated_moves.base_evaluation = eval_board(board, white_to_play);

    // check if the game is over
    if (tlm_struct.outcome != 0) {

        // it is game over, there are no moves to find
        generated_moves.base_evaluation = tlm_struct.evaluation;
        generated_moves.game_continues = false;
        return generated_moves;
    }
    else {
        generated_moves.game_continues = true;
    }

    // loop through all of the legal moves
    int loop_num = tlm_struct.legal_moves.size() / 3;
    for (int i = 0; i < loop_num; i++) {

        int start_sq = tlm_struct.legal_moves[(i * 3) + 0];
        int dest_sq = tlm_struct.legal_moves[(i * 3) + 1];
        int move_mod = tlm_struct.legal_moves[(i * 3) + 2];

        // find the indexes required for the tlm data array
        int start_ind = (start_sq - 21) - 2 * ((start_sq / 10) - 2);
        int dest_ind = (dest_sq - 21) - 2 * ((dest_sq / 10) - 2);

        // has the destination square not already got piece view filled in
        if (tlm_struct.square_info[dest_ind].piece_view.size() == 0) {
            // we need to find the piece view (any piece/colour will do)
            temp_pad_struct = piece_attack_defend(board, dest_sq, 6, 1);
            piece_view = temp_pad_struct.piece_view;
            //piece_view = find_view(board, dest_sq);
        }
        else {
            piece_view = tlm_struct.square_info[dest_ind].piece_view;
        }

        // add the destination square itself to the piece view
        piece_view.push_back(dest_sq);

        // now we add the piece view of the start square (without duplicates)
        for (int v : tlm_struct.square_info[start_ind].piece_view) {
            if (not is_in(piece_view, v)) {
                piece_view.push_back(v);
            }
        }

        //// FOR TESTING
        //std::cout << "piece view is ";
        //for (int x : piece_view) {
        //    std::cout << x << ", ";
        //}
        //std::cout << " start_sq is " << start_sq << "; dest_sq is " << dest_sq;
        //std::cout << "\n";

        // if it is a capture of a pawn, or by a pawn, we may create a passed pawn
        if (move_mod == 2) {

            int pawn_step = 0;
            int step_start[2];

            // if a pawn is capturing, its movement may create enemy passed pawns
            if (board.arr[start_sq] == 1 or board.arr[start_sq] == -1) {

                // which way to step to find enemy pawns
                if (next_to_play) pawn_step = -10;
                else pawn_step = 10;

                // which columns to step along
                if (dest_sq > start_sq) {
                    step_start[0] = start_sq - 1;
                    step_start[1] = dest_sq + 1;
                }
                else {
                    step_start[0] = start_sq + 1;
                    step_start[1] = dest_sq - 1;
                }

                for (int step : step_start) {
                    if (board.arr[step] != 7 and board.arr[step] != -7) {
                        for (int i = 0; i < 7; i++) {
                            step += pawn_step;
                            if (board.arr[step] == 1 or board.arr[step] == -1) {
                                // if we find a pawn, finish searching
                                if (not is_in(piece_view, step)) {
                                    piece_view.push_back(step);

                                    //// FOR TESTINGG
                                    //std::cout << "The move will be " << start_sq
                                    //    << " -> " << dest_sq << " (" << move_mod
                                    //    << ")\n";
                                    //std::cout << "We have added the pawn on square "
                                    //    << step << " to the board:\n";
                                    //print_board(board, false);
                                }
                                break;
                            }
                            else if (board.arr[step] == -7 or
                                board.arr[step] == 7) {
                                break;
                            }
                        }
                    }
                }
            }

            // if a pawn is being captured, we may create our own passed pawns
            if (board.arr[dest_sq] == 1 or board.arr[dest_sq] == -1) {

                // which way to step to find our pawns
                if (next_to_play) pawn_step = 10;
                else pawn_step = -10;

                // which columns to step along
                step_start[0] = dest_sq - 1;
                step_start[1] = dest_sq + 1;

                for (int step : step_start) {
                    if (board.arr[step] != 7 and board.arr[step] != -7) {
                        for (int i = 0; i < 7; i++) {
                            step += pawn_step;
                            if (board.arr[step] == 1 or board.arr[step] == -1) {
                                // if we find a pawn, finish searching
                                if (not is_in(piece_view, step)) {
                                    piece_view.push_back(step);

                                    //// FOR TESTINGG
                                    //std::cout << "The move will be " << start_sq
                                    //    << " -> " << dest_sq << " (" << move_mod
                                    //    << ")\n";
                                    //std::cout << "We have added the pawn on square "
                                    //    << step << " to the board:\n";
                                    //print_board(board, false);
                                }
                                break;
                            }
                            else if (board.arr[step] == -7 or
                                board.arr[step] == 7) {
                                break;
                            }
                        }
                    }
                }
            }
        }

        // if it is capture en passant, ensure we include the captured square
        if (move_mod == 3) {
            if (white_to_play) {
                if (not is_in(piece_view, dest_sq - 10)) {
                    piece_view.push_back(dest_sq - 10);
                }
            }
            else {
                if (not is_in(piece_view, dest_sq + 10)) {
                    piece_view.push_back(dest_sq + 10);
                }
            }
        }

        // castling
        if (move_mod == 5) {
            // where does the rook end up
            int rook_sq;
            // queenside castle
            if (dest_sq > start_sq) {
                rook_sq = dest_sq - 1;
            }
            // kingside castle
            else {
                rook_sq = dest_sq + 1;
            }

            // we need to find the piece view of the rook
            temp_pad_struct = piece_attack_defend(board, rook_sq, 6, 1);

            // add the piece view of the rook after moving (without duplicates)
            for (int v : temp_pad_struct.piece_view) {
                if (not is_in(piece_view, v)) {
                    piece_view.push_back(v);
                }
            }

            piece_view.push_back(rook_sq);
        }

        // copy the base board
        Board new_board = board;

        // make the move on the board copy
        make_move(new_board, start_sq, dest_sq, move_mod);

        // make a copy of the board evaluation on the next move
        int new_eval = tlm_struct.evaluation;

        // find the new phase
        phase_struct phase = determine_phase(new_board, next_to_play);

        /* This does fix phase transition cases, in testing phase errors accounted
        for about 30 / 7200 errors (0.4%). It may be that these are acceptable
        errors for the speed boost and potential to avoid moves that change phase
        being unfairly favoured (eg to reduce a non-castled penalty) */
        // FOR TESTING: COMPLETE RE-EVAL IF PHASE CHANGES
        if (phase.phase != tlm_struct.phase) {
            // create a move entry
            move_struct new_move;

            // fill in the data fields for this new move
            new_move.board = new_board;
            new_move.start_sq = start_sq;
            new_move.dest_sq = dest_sq;
            new_move.move_mod = move_mod;
            new_move.evaluation = eval_board(new_board, next_to_play);
            //new_move.hash = ...

            // insert the new move into the generated_moves vector
            ordered_insert_moves(new_move, generated_moves.moves, white_to_play);

            continue;
        }

        // set how much we value creating threats
        phase.null_active = phase.null_attack;

        // update the new evaluation based on phase ajust change
        new_eval += phase.evaluation_adjust - tlm_struct.phase_adjust;
        //new_eval += phase.evaluation_adjust - old_phase.evaluation_adjust;

        // loop through view squares and recalculate their value
        for (int view : piece_view) {

            // what was the old piece evaluation
            int view_ind = (view - 21) - 2 * ((view / 10) - 2);
            int old_value = tlm_struct.square_info[view_ind].evaluation;
            int new_value;

            // extract details of piece on this square now
            int piece_type = new_board.arr[view];
            int piece_colour;
            if (piece_type < 0) {
                piece_type *= -1;
                piece_colour = -1;
            }
            else if (piece_type > 0) {
                piece_colour = 1;
            }

            // now investigate this piece
            if (piece_type == 0) {
                new_value = 0;
            }
            else {
                // find the new piece information
                temp_pad_struct = piece_attack_defend(new_board, view,
                    piece_type, piece_colour);

                // find the new piece evaluation
                new_value = eval_piece(new_board, next_to_play, view,
                    phase, piece_type, piece_colour, mate, temp_pad_struct);

                // have we found a checkmate
                if (mate) {

                    //// FOR TESTING
                    // verify_checkmate(new_board, next_to_play);
                    //std::cout << "Found a mate!!!!!!!!!!!!!!!\n";
                    //std::cout << "The move was: " << start_sq << " -> " << dest_sq
                    //    << " (" << move_mod << ")\n";
                    //std::cout << "The new board is:\n";
                    //print_board(new_board, false);

                    generated_moves.mating_move = true;
                    if (next_to_play) {
                        new_eval = WHITE_MATED;
                    }
                    else {
                        new_eval = BLACK_MATED;
                    }
                    // no need to continue with this board
                    break;
                }
            }

            // we have the value of this square, update the board evaluation
            new_eval += new_value - old_value;
        }

        // we now have the new evaluation of this board, following the move

        // create a move entry
        move_struct new_move;

        // fill in the data fields for this new move
        new_move.board = new_board;
        new_move.start_sq = start_sq;
        new_move.dest_sq = dest_sq;
        new_move.move_mod = move_mod;
        new_move.evaluation = new_eval;
        //new_move.hash = ...

        // insert the new move into the generated_moves vector
        ordered_insert_moves(new_move, generated_moves.moves, white_to_play);
    }

    // we have now gone through every legal move
    return generated_moves;
}

bool is_promotion(Board& board, bool white_to_play, std::string move_letters)
{
    /* check if a move is a promotion */

    // confirm the move is the correct length
    if (move_letters.size() != 4) {
        if (move_letters.size() != 5 and
            (move_letters[4] != 'n' and
                move_letters[4] != 'q' and
                move_letters[4] != 'b' and
                move_letters[4] != 'r')) {
            throw std::runtime_error("move length != 4 or 5, bad input");
        }
    }

    int start_sq = square_letters_to_numbers(move_letters.substr(0, 2));
    int dest_sq = square_letters_to_numbers(move_letters.substr(2, 2));

    if (board.arr[start_sq] == 1 and white_to_play) {
        if (start_sq / 10 == 8 and dest_sq / 10 == 9) {
            return true;
        }
    }
    else if (board.arr[start_sq] == -1 and not white_to_play) {
        if (start_sq / 10 == 3 and dest_sq / 10 == 2) {
            return true;
        }
    }

    return false;
}

verified_move verify_move(Board& board, bool white_to_play, std::string move_letters)
{
    /* check if a move is legal from its letters */

    verified_move move;
    move.legal = false;
    move.start_sq = 0;
    move.dest_sq = 0;
    move.move_mod = -1;

    // confirm the move is the correct length
    if (move_letters.size() != 4) {
        if (move_letters.size() != 5 and
            (move_letters[4] != 'n' and
                move_letters[4] != 'q' and
                move_letters[4] != 'b' and
                move_letters[4] != 'r')) {
            std::cout << "Input move does not have the correct length"
                << ", it should be of the form e2e4, d7d5, d7d8q etc\n";
            return move;
        }
    }

    // confirm that the characters are in the correct ranges
    if (move_letters[0] < 97 or move_letters[0] > 104 or
        move_letters[1] < 49 or move_letters[1] > 56 or
        move_letters[2] < 97 or move_letters[2] > 104 or
        move_letters[3] < 49 or move_letters[3] > 56) {
        std::cout << "Input move characters do not fit within the "
            << "expected range, a-h and 1-8. It should be of the form"
            << " e2e4, d7d5, d7d8q etc, no capitals\n";
        return move;
    }

    std::string num_string;

    // convert characters to numbers
    for (int i = 0; i < 2; i++) {

        char y = move_letters[i * 2];
        char x = move_letters[i * 2 + 1];

        num_string += x + 1; // '1' becomes '2' etc
        num_string += 153 - y; // 'h' becomes '1' etc

    }

    // convert string to numbers
    int start_sq = std::stoi(num_string.substr(0, 2));
    int dest_sq = std::stoi(num_string.substr(2, 2));

    move = verify_move(board, white_to_play, start_sq, dest_sq);

    // if a promotion choice has been given

    // if the move is a promotion
    if (move.move_mod == 10) {
        if (move_letters.size() == 5) {
            switch (move_letters[4]) {
            case 'n': move.move_mod = 6; break;
            case 'q': move.move_mod = 7; break;
            case 'b': move.move_mod = 8; break;
            case 'r': move.move_mod = 9; break;
            }
        }
        else {
            // default is queen promote
            move.move_mod = 7;
        }
    }

    return move;
}

verified_move verify_move(Board& board, bool white_to_play, int start_sq, int dest_sq)
{
    /* check if a move is legal on a given board */

    total_legal_moves_struct tlm_struct = total_legal_moves(board, white_to_play);

    verified_move move;
    move.legal = false;
    move.start_sq = 0;
    move.dest_sq = 0;
    move.move_mod = -1;

    // see if the proposed move is legal
    int loops = tlm_struct.legal_moves.size() / 3;
    for (int i = 0; i < loops; i++) {

        int st = tlm_struct.legal_moves[i * 3];
        int ds = tlm_struct.legal_moves[i * 3 + 1];
        int mv = tlm_struct.legal_moves[i * 3 + 2];

        if (st == start_sq and ds == dest_sq) {
            move.legal = true;
            move.start_sq = st;
            move.dest_sq = ds;
            move.move_mod = mv;
            break;
        }
    }

    // if the move is promotion, set move_mod=10 to indicate this
    if (move.move_mod == 6 or
        move.move_mod == 7 or
        move.move_mod == 8 or
        move.move_mod == 9) {
        move.move_mod = 10;
    }

    return move;
}

Board create_board(std::vector<std::string> moves)
{
    /* This function creates a board from a given list of moves */

    // first, start with an empty board
    Board board = create_board();
    bool white_to_play = true;

    // now loop through the moves, and apply them to the board
    for (std::string& move : moves) {

        // check the move is legal, then if so, make it on the board
        verified_move move_numbers = verify_move(board, white_to_play, move);
        if (move_numbers.legal) {
            make_move(board, move_numbers.start_sq, move_numbers.dest_sq,
                move_numbers.move_mod);
        }
        else {
            std::string error_msg = "The move " + move + " is not legal!";
            throw std::runtime_error(error_msg);
        }

        white_to_play = not white_to_play;
    }

    return board;
}

std::string get_game_outcome(Board& board, bool white_to_play)
{
    total_legal_moves_struct tlm = total_legal_moves(board, white_to_play);

    if (tlm.outcome == 0) {
        return "continue";
    }
    else if (tlm.outcome == 1) {
        return "checkmate";
    }
    else if (tlm.outcome == 2) {
        return "stalemate";
    }
    else {
        throw std::runtime_error("tlm_struct.outcome is out of bounds (not 0,1,2)");
    }
}

bool are_boards_identical(Board first, Board second)
{
    /* This function determines if two boards are exactly the same */

    for (int i = 0; i < 120; i++) {

        // ignore the exceptions
        //if (i == 4) continue;       // the passant wipe boolean
        if (i == 100) continue;     // phase flag


        if (first.arr[i] != second.arr[i]) {
            std::cout << "Board difference at index " << i << ". First board "
                << first.arr[i] << ", second board " << second.arr[i] << "\n";
            return false;
        }
    }

    return true;
}

Board copy_board(Board& base_board)
{ 
    /* This function returns a copy of a base board */

    Board new_board;

    for (int i = 0; i < 120; i++) {
        new_board.arr[i] = base_board.arr[i];
    }

    return new_board;
}

//int main()
//{
//    // create a board
//    //int board[120];
//
//    int board[120];
//
//    create_board_a(board);
//
//    //// lets test the castling behaviour
//    //board[22] = 0;
//    //board[23] = 0;
//    //board[52] = -5;
//    //board[42] = -3;
//    //board[32] = 0;
//    //board[5] = 7;
//    //std::vector<int> ls_castles = list_of_castles(board, 1);
//
//    //// print the output
//    //std::cout << "The legal castles are: ";
//    //for (int x : ls_castles) {
//    //    std::cout << " " << x;
//    //}
//    //std::cout << '\n';
//
//    move_piece_on_board_a(board, 33, 53);
//
//    // print the board
//    print_board_a(board,false);
//
//    /*move_piece_on_board(board, 85, 65);
//    print_board(board, false);
//
//    move_piece_on_board(board, 34, 44);
//    print_board(board, false);
//
//    move_piece_on_board(board, 86, 66);
//    print_board(board, false);
//    move_piece_on_board(board, 44, 54);
//    print_board(board, false);*/
//
//    /*int z = 0;
//    for (int i = 0; i < 120; i++) {
//        if (z % 10 == 0) {
//            std::cout << '\n';
//        }
//        std::cout << " " << *(board_ptr + i);
//        z++;
//    }*/
//    
//    /*for (int y : board) {
//        if (z % 10 == 0) {
//            std::cout << '\n';
//        }
//        std::cout << " " << y;
//        ++z;
//    }*/
//
//    
//
//}

// lets test out some pybind stuff

// namespace py = pybind11;

// PYBIND11_MODULE(board_functions, m) {

//     // data structures
//     py::class_<Board>(m, "Board")
//         .def(py::init<>())
//         .def("look", &Board::look)
//         .def("set", &Board::set);

//     py::class_<piece_attack_defend_struct>(m, "piece_attack_defend_struct")
//         .def(py::init<>())
//         .def("get_attack_list", &piece_attack_defend_struct::get_attack_list)
//         .def("get_attack_me", &piece_attack_defend_struct::get_attack_me)
//         .def("get_defend_me", &piece_attack_defend_struct::get_defend_me)
//         .def("get_piece_view", &piece_attack_defend_struct::get_piece_view)
//         .def("get_evalution", &piece_attack_defend_struct::get_evaluation);

//     py::class_<total_legal_moves_struct>(m, "total_legal_moves_struct")
//         .def(py::init<>())
//         .def("get_square", &total_legal_moves_struct::get_square)
//         .def("get_from_index", &total_legal_moves_struct::get_from_index)
//         .def("set_from_index", &total_legal_moves_struct::set_from_index)
//         .def("get_legal_moves", &total_legal_moves_struct::get_legal_moves)
//         .def("get_outcome", &total_legal_moves_struct::get_outcome)
//         .def("get_evaluation", &total_legal_moves_struct::get_evaluation)
//         .def("get_phase", &total_legal_moves_struct::get_phase)
//         .def("get_phase_adjust", &total_legal_moves_struct::get_phase_adjust)
//         .def("get_old_value", &total_legal_moves_struct::get_old_value)
//         .def("get_piece_view", &total_legal_moves_struct::get_piece_view)
//         ;

//     py::class_<phase_struct>(m, "phase_struct")
//         .def(py::init<>())
//         .def("get_eval_adjust", &phase_struct::get_eval_adjust)
//         .def("get_phase", &phase_struct::get_phase)
//         ;

//     py::class_<move_struct>(m, "move_struct")
//         .def(py::init<>())
//         .def("get_evaluation", &move_struct::get_evaluation)
//         .def("get_hash", &move_struct::get_hash)
//         .def("get_start_sq", &move_struct::get_start_sq)
//         .def("get_dest_sq", &move_struct::get_dest_sq)
//         .def("get_move_mod", &move_struct::get_move_mod)
//         .def("get_board", &move_struct::get_board);

//     py::class_<generated_moves_struct>(m, "generated_moves_struct")
//         .def(py::init<>())
//         .def("get_evaluation", &generated_moves_struct::get_evaluation)
//         .def("get_outcome", &generated_moves_struct::get_outcome)
//         .def("get_board", &generated_moves_struct::get_board)
//         .def("get_moves", &generated_moves_struct::get_moves)
//         .def("does_game_continue", &generated_moves_struct::does_game_continue)
//         .def("get_length", &generated_moves_struct::get_length)
//         .def("is_mating_move", &generated_moves_struct::is_mating_move);

//     // functions that use the board struct
//     m.def("create_board", py::overload_cast<>(&create_board));
//     m.def("create_board", py::overload_cast<std::vector<std::string>>(&create_board));
//     m.def("print_board", py::overload_cast<Board&>(&print_board));
//     m.def("print_board", py::overload_cast<Board&, bool>(&print_board));
//     m.def("py_print_board", py::overload_cast<Board&>(&py_print_board));
//     m.def("py_print_board", py::overload_cast<Board&, bool>(&py_print_board));
//     m.def("move_piece_on_board", &move_piece_on_board);
//     m.def("is_in_check", &is_in_check);
//     m.def("list_of_castles", &list_of_castles);
//     m.def("make_move", &make_move);
//     m.def("piece_attack_defend", &piece_attack_defend);
//     m.def("total_legal_moves", &total_legal_moves);
//     m.def("find_pins_checks", &find_pins_checks);
//     m.def("king_walks", &king_walks);
//     m.def("is_in", &is_in);
//     m.def("get_my_pin_moves", &get_my_pin_moves);
//     m.def("is_checkmate", &is_checkmate);
//     m.def("test_checkmate", &test_checkmate);
//     m.def("tempo_check", &tempo_check);
//     m.def("linear_insert", &linear_insert);
//     m.def("order_attackers_defenders", &order_attackers_defenders);
//     m.def("determine_phase", &determine_phase);
//     m.def("piece_value", &piece_value);
//     m.def("eval_piece", &eval_piece);
//     m.def("eval_board", &eval_board);
//     m.def("ordered_insert_moves", &ordered_insert_moves);
//     m.def("generate_moves", &generate_moves);
//     m.def("set_evaluation_settings", &set_evaluation_settings);
//     m.def("are_boards_identical", &are_boards_identical);
//     m.def("copy_board", &copy_board);
//     m.def("find_view", &find_view);
//     m.def("get_game_outcome", &get_game_outcome);
//     m.def("square_letters_to_numbers", &square_letters_to_numbers);
//     m.def("square_numbers_to_letters", &square_numbers_to_letters);
//     m.def("is_promotion", &is_promotion);

// #ifdef VERSION_INFO
//     m.attr("__version__") = VERSION_INFO;
// #else
//     m.attr("__version__") = "dev";
// #endif
// }

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file