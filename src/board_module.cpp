#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "board_functions.h"

namespace py = pybind11;

PYBIND11_MODULE(board_module, m) {

    // data structures
    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def("look", &Board::look)
        .def("set", &Board::set)
        ;

    py::class_<piece_attack_defend_struct>(m, "piece_attack_defend_struct")
        .def(py::init<>())
        .def("get_attack_list", &piece_attack_defend_struct::get_attack_list)
        .def("get_attack_me", &piece_attack_defend_struct::get_attack_me)
        .def("get_defend_me", &piece_attack_defend_struct::get_defend_me)
        .def("get_piece_view", &piece_attack_defend_struct::get_piece_view)
        .def("get_evalution", &piece_attack_defend_struct::get_evaluation)
        ;

    py::class_<total_legal_moves_struct>(m, "total_legal_moves_struct")
        .def(py::init<>())
        .def("get_square", &total_legal_moves_struct::get_square)
        .def("get_from_index", &total_legal_moves_struct::get_from_index)
        .def("set_from_index", &total_legal_moves_struct::set_from_index)
        .def("get_legal_moves", &total_legal_moves_struct::get_legal_moves)
        .def("get_outcome", &total_legal_moves_struct::get_outcome)
        .def("get_evaluation", &total_legal_moves_struct::get_evaluation)
        .def("get_phase", &total_legal_moves_struct::get_phase)
        .def("get_phase_adjust", &total_legal_moves_struct::get_phase_adjust)
        .def("get_old_value", &total_legal_moves_struct::get_old_value)
        .def("get_piece_view", &total_legal_moves_struct::get_piece_view)
        ;

    py::class_<phase_struct>(m, "phase_struct")
        .def(py::init<>())
        .def("get_eval_adjust", &phase_struct::get_eval_adjust)
        .def("get_phase", &phase_struct::get_phase)
        ;

    py::class_<move_struct>(m, "move_struct")
        .def(py::init<>())
        .def("get_evaluation", &move_struct::get_evaluation)
        .def("get_hash", &move_struct::get_hash)
        .def("get_start_sq", &move_struct::get_start_sq)
        .def("get_dest_sq", &move_struct::get_dest_sq)
        .def("get_move_mod", &move_struct::get_move_mod)
        .def("get_board", &move_struct::get_board)
        .def("to_letters", &move_struct::to_letters)
        ;

    py::class_<generated_moves_struct>(m, "generated_moves_struct")
        .def(py::init<>())
        .def("get_evaluation", &generated_moves_struct::get_evaluation)
        .def("get_outcome", &generated_moves_struct::get_outcome)
        .def("get_board", &generated_moves_struct::get_board)
        .def("get_moves", &generated_moves_struct::get_moves)
        .def("does_game_continue", &generated_moves_struct::does_game_continue)
        .def("get_length", &generated_moves_struct::get_length)
        .def("is_mating_move", &generated_moves_struct::is_mating_move)
        ;

    py::class_<BoardVectors>(m, "BoardVectors")
        .def(py::init<>())
        .def_readwrite("wP", &BoardVectors::wP)
        .def_readwrite("wN", &BoardVectors::wN)
        .def_readwrite("wB", &BoardVectors::wB)
        .def_readwrite("wR", &BoardVectors::wR)
        .def_readwrite("wQ", &BoardVectors::wQ)
        .def_readwrite("wK", &BoardVectors::wK)
        .def_readwrite("bP", &BoardVectors::bP)
        .def_readwrite("bN", &BoardVectors::bN)
        .def_readwrite("bB", &BoardVectors::bB)
        .def_readwrite("bR", &BoardVectors::bR)
        .def_readwrite("bQ", &BoardVectors::bQ)
        .def_readwrite("bK", &BoardVectors::bK)
        .def_readwrite("wKS", &BoardVectors::wKS)
        .def_readwrite("wQS", &BoardVectors::wQS)
        .def_readwrite("bKS", &BoardVectors::bKS)
        .def_readwrite("bQS", &BoardVectors::bQS)
        .def_readwrite("colour", &BoardVectors::colour)
        .def_readwrite("total_moves", &BoardVectors::total_moves)
        .def_readwrite("no_take_ply", &BoardVectors::no_take_ply)
        ;

    // expose only functions we want to be usable from python
    m.def("generate_moves_FEN", &generate_moves_FEN);
    m.def("print_FEN_board", &print_FEN_board);
    m.def("FEN_to_board_vectors", &FEN_to_board_vectors);
    m.def("is_white_next_FEN", &is_white_next_FEN);
    m.def("FEN_and_move_to_board_vectors", &FEN_and_move_to_board_vectors);

    // functions that use the board struct
    // m.def("create_board", py::overload_cast<>(&create_board));
    // m.def("create_board", py::overload_cast<std::vector<std::string>>(&create_board));
    // m.def("print_board", py::overload_cast<Board&>(&print_board));
    // m.def("print_board", py::overload_cast<Board&, bool>(&print_board));
    // m.def("py_print_board", py::overload_cast<Board&>(&py_print_board));
    // m.def("py_print_board", py::overload_cast<Board&, bool>(&py_print_board));
    // m.def("move_piece_on_board", &move_piece_on_board);
    // m.def("is_in_check", &is_in_check);
    // m.def("list_of_castles", &list_of_castles);
    // m.def("make_move", &make_move);
    // m.def("piece_attack_defend", &piece_attack_defend);
    // m.def("total_legal_moves", &total_legal_moves);
    // m.def("find_pins_checks", &find_pins_checks);
    // m.def("king_walks", &king_walks);
    // m.def("is_in", &is_in);
    // m.def("get_my_pin_moves", &get_my_pin_moves);
    // m.def("is_checkmate", &is_checkmate);
    // m.def("test_checkmate", &test_checkmate);
    // m.def("tempo_check", &tempo_check);
    // m.def("linear_insert", &linear_insert);
    // m.def("order_attackers_defenders", &order_attackers_defenders);
    // m.def("determine_phase", &determine_phase);
    // m.def("piece_value", &piece_value);
    // m.def("eval_piece", &eval_piece);
    // m.def("eval_board", &eval_board);
    // m.def("ordered_insert_moves", &ordered_insert_moves);
    // m.def("generate_moves", &generate_moves);
    // m.def("set_evaluation_settings", &set_evaluation_settings);
    // m.def("are_boards_identical", &are_boards_identical);
    // m.def("copy_board", &copy_board);
    // m.def("find_view", &find_view);
    // m.def("get_game_outcome", &get_game_outcome);
    // m.def("square_letters_to_numbers", &square_letters_to_numbers);
    // m.def("square_numbers_to_letters", &square_numbers_to_letters);
    // m.def("is_promotion", &is_promotion);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}