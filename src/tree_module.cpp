#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "tree_functions.h"

namespace py = pybind11;

PYBIND11_MODULE(tree_module, m) {

    // data structures
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def(py::init<std::string&>())
        .def("get_start_sq", &Move::get_start_sq)
        .def("get_dest_sq", &Move::get_dest_sq)
        .def("get_move_mod", &Move::get_move_mod)
        .def("to_letters", &Move::to_letters)
        .def("print", &Move::print)
        .def("set", &Move::set)
        ;

    py::class_<TreeKey>(m, "TreeKey")
        .def(py::init<>())
        .def("get_layer", &TreeKey::get_layer)
        .def("get_entry", &TreeKey::get_entry)
        .def("get_move_index", &TreeKey::get_move_index)
        .def("get_evaluation", &TreeKey::get_evaluation)
        .def("print", &TreeKey::print)
        ;

    py::class_<MoveEntry>(m, "MoveEntry")
        .def(py::init<>())
        .def("get_move", &MoveEntry::get_move)
        .def("get_new_eval", &MoveEntry::get_new_eval)
        .def("get_new_hash", &MoveEntry::get_new_hash)
        .def("print", py::overload_cast<>(&MoveEntry::print))
        .def("print", py::overload_cast<std::string>(&MoveEntry::print))
        .def("to_letters", &MoveEntry::to_letters)
        .def("get_depth_evaluated", &MoveEntry::get_depth_evaluated)
        ;

    py::class_<TreeEntry>(m, "TreeEntry")
        //.def(py::init<>())
        .def(py::init<int>())
        .def("print", py::overload_cast<>(&TreeEntry::print))
        .def("print", py::overload_cast<std::string>(&TreeEntry::print))
        .def("print", py::overload_cast<bool>(&TreeEntry::print))
        .def("print", py::overload_cast<bool, std::string>(&TreeEntry::print))
        .def("get_hash_key", &TreeEntry::get_hash_key)
        .def("get_move_from_list", &TreeEntry::get_move_from_list)
        .def("get_parent_keys", &TreeEntry::get_parent_keys)
        .def("get_board_state", &TreeEntry::get_board_state)
        .def("get_eval", &TreeEntry::get_eval)
        .def("is_active", &TreeEntry::is_active)
        ;

    py::class_<TreeLayer>(m, "TreeLayer")
        //.def(py::init<>())
        .def(py::init<int, int, int, bool>())
        .def("add", &TreeLayer::add)
        .def("find_hash", &TreeLayer::find_hash)
        .def("print", &TreeLayer::print)
        .def("get_hash_from_list", &TreeLayer::get_hash_from_list)
        .def("get_key_from_list", &TreeLayer::get_key_from_list)
        ;

    py::class_<LayeredTree>(m, "LayeredTree")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<Board, bool, int>())
        .def("set_root", py::overload_cast<>(&LayeredTree::set_root))
        .def("set_root", py::overload_cast<Board, bool>(&LayeredTree::set_root))
        .def("get_boards_checked", &LayeredTree::get_boards_checked)
        .def("add_layer", &LayeredTree::add_layer)
        .def("remove_layer", &LayeredTree::remove_layer)
        .def("set_width", &LayeredTree::set_width)
        .def("print", py::overload_cast<>(&LayeredTree::print))
        .def("print", py::overload_cast<int>(&LayeredTree::print))
        .def("print_old_ids", &LayeredTree::print_old_ids)
        .def("print_new_ids", py::overload_cast<>(&LayeredTree::print_new_ids))
        .def("print_new_ids", py::overload_cast<int>(&LayeredTree::print_new_ids))
        .def("add_element", &LayeredTree::add_move)
        .def("add_board_replies", &LayeredTree::add_board_replies)
        .def("grow_tree", &LayeredTree::grow_tree)
        .def("advance_ids", &LayeredTree::advance_ids)
        .def("cascade", &LayeredTree::cascade)
        .def("limit_prune", &LayeredTree::limit_prune)
        .def("print_best_move", &LayeredTree::print_best_move)
        .def("print_best_moves", py::overload_cast<>(&LayeredTree::print_best_moves))
        .def("print_best_moves", py::overload_cast<TreeKey>(&LayeredTree::print_best_moves))
        .def("print_best_moves", py::overload_cast<bool>(&LayeredTree::print_best_moves))
        .def("print_best_moves", py::overload_cast<TreeKey, bool>(&LayeredTree::print_best_moves))
        .def("print_boards_checked", &LayeredTree::print_boards_checked)
        .def("next_layer", py::overload_cast<>(&LayeredTree::next_layer))
        .def("next_layer", py::overload_cast<int, int>(&LayeredTree::next_layer))
        .def("target_prune", &LayeredTree::target_prune)

        .def("print_id_list", &LayeredTree::print_id_list)
        .def("test_dictionary", &LayeredTree::test_dictionary)
        .def("find_hash", &LayeredTree::find_hash)
        ;

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("set_width", &Engine::set_width)
        .def("set_depth", &Engine::set_depth)
        .def("generate_engine_moves_FEN", &Engine::generate_engine_moves_FEN, 
            py::arg("fen_string") = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", py::arg("target_time") = 5)
        ;

    py::class_<GameBoard>(m, "GameBoard")
        .def(py::init<>())
        .def(py::init<Board, bool>())
        .def(py::init<std::vector<std::string>>())
        .def("get_board", &GameBoard::get_board)
        .def("get_white_to_play", &GameBoard::get_white_to_play)
        .def("get_move_list", &GameBoard::get_move_list)
        .def("get_last_move", &GameBoard::get_last_move)
        .def("get_outcome", &GameBoard::get_outcome)
        .def("check_promotion", &GameBoard::check_promotion)
        .def("get_square_colour", &GameBoard::get_square_colour)
        .def("get_square_raw_value",
            py::overload_cast<int>(&GameBoard::get_square_raw_value))
        .def("get_square_raw_value",
            py::overload_cast<std::string>(&GameBoard::get_square_raw_value))
        .def("get_square_piece", py::overload_cast<int>(&GameBoard::get_square_piece))
        .def("get_square_piece",
            py::overload_cast<std::string>(&GameBoard::get_square_piece))
        .def("move", &GameBoard::move)
        .def("undo", &GameBoard::undo)
        .def("reset", py::overload_cast<>(&GameBoard::reset))
        .def("reset", py::overload_cast<std::vector<std::string>>(&GameBoard::reset))
        .def("reset", py::overload_cast<Board, bool>(&GameBoard::reset))
        .def("get_engine_move", py::overload_cast<>(&GameBoard::get_engine_move))
        .def("get_engine_move", py::overload_cast<int>(&GameBoard::get_engine_move))
        .def("get_engine_move_no_GIL", &GameBoard::get_engine_move_no_GIL,
            py::call_guard<py::gil_scoped_release>())
        ;

    //m.def("get_engine_move_no_GIL", [](GameBoard* gameboard) -> std::string {
    //    // release GIL first
    //    py::gil_scoped_release release;
    //    return get_engine_move_no_GIL(gameboard);
    //    });

    py::class_<Game>(m, "Game")
        .def(py::init<>())
        .def("play_terminal", &Game::play_terminal)
        .def("is_move_legal", &Game::is_move_legal)
        .def("human_move", &Game::human_move)
        .def("engine_move", &Game::engine_move)
        .def("get_last_move", &Game::get_last_move)
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}