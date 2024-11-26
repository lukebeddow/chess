// Tree functions.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "tree_functions.h"

#if defined(LUKE_PYBIND)
    namespace py = pybind11;
    using namespace pybind11::literals; // for py::print
#endif

void print_str(std::string s, bool newline = true)
{
    /* special function to seperate pybind printing from c++ only printing */

    #if defined(LUKE_PYBIND)
        if (newline) py::print(s);
        else py::print(s, "end"_a = "");
    #else
        std::cout << s;
        if (newline) std::cout << "\n";
    #endif
}

void print_vector(std::vector<int> v, std::string name) {
    std::cout << name << ": ";
    for (int i = 0; i < v.size() - 1; i++) {
        std::cout << v[i] << ", ";
    }
    std::cout << v[v.size() - 1] << ".\n";
}

void print_vector(std::vector<TreeKey> v, std::string name) {
    std::cout << name << ":\n";
    for (TreeKey& t : v) {
        t.print();
    }
}

void print_vector(std::vector<std::string> v, std::string name)
{
    std::cout << name << ": ";
    for (int i = 0; i < v.size() - 1; i++) {
        std::cout << v[i] << ", ";
    }
    std::cout << v[v.size() - 1] << ".\n";
}

Move::Move(std::string move_letters)
{
    /* Constructor to make a move using letters */

    if (move_letters.size() != 4) {
        if (move_letters.size() != 5 and
               (move_letters[4] != 'n' and
                move_letters[4] != 'q' and
                move_letters[4] != 'b' and
                move_letters[4] != 'r')) {
            throw std::runtime_error("incorrect move length, should be 4 characters (or 5 if promotion)");
        }
    }

    std::string num_string;

    for (int i = 0; i < 2; i++) {

        char y = move_letters[i * 2];
        char x = move_letters[i * 2 + 1];

        num_string += x + 1; // '1' becomes '2' etc
        num_string += 153 - y; // 'h' becomes '1' etc
        
    }

    start_sq = std::stoi(num_string.substr(0, 2));
    dest_sq = std::stoi(num_string.substr(2, 2));

    if (move_letters.size() == 5) {
        switch (move_letters[4]) {
        case 'n': move_mod = 6; break;
        case 'q': move_mod = 7; break;
        case 'b': move_mod = 8; break;
        case 'r': move_mod = 9; break;
        }
    }
}

std::string Move::to_letters()
{
    /* This function converts a move from numbers to letters */

    // create strings
    std::string move_letters;
    std::string ind_letters = std::to_string(start_sq) + std::to_string(dest_sq);
    
    if (ind_letters.size() != 4) {
        //throw std::runtime_error("move string doesn't contain 4 letters! Incorrect move");
        return "error";
    }

    for (int i = 0; i < 2; i++) {

        // what are the two numbers that make up this square
        char x = ind_letters[2 * i];
        char y = ind_letters[2 * i + 1];

        move_letters += 153 - y; // '1' becomes 'h' etc
        move_letters += x - 1; // '2' becomes '1' etc

    }

    // if we are promoting, add clarification
    switch (move_mod) {
    case 6: move_letters += 'n'; break;
    case 7: move_letters += 'q'; break;
    case 8: move_letters += 'b'; break;
    case 9: move_letters += 'r'; break;
    }

    return move_letters;
}

void Move::print()
{
    std::cout << "Move | start sq = " << start_sq << ", dest_sq = " << dest_sq
        << ", move_mod = " << move_mod << " (letters = " << to_letters()
        << ")\n";
}

void TreeKey::print()
{
    std::cout << "TreeKey | layer = " << layer << ", entry = " << entry
        << ", move_index = " << move_index << ", evaluation = "
        << evaluation << '\n';
}

void MoveEntry::print()
{
    print("");
}

void MoveEntry::print(std::string starter)
{
    std::cout << starter + "MoveEntry:" << '\n';
    std::cout << starter + '\t';
    move.print();
    std::cout << starter + '\t' + "new_eval = " << new_eval << " ("
        << new_eval / 1000.0 << "), new_hash = " << new_hash
        << ", active_move = " << active_move << '\n';
}

std::string MoveEntry::print_move()
{
    /* get the move as a string */

    return move.to_letters();
}

std::string MoveEntry::print_eval()
{
    /* get the evaluation of the node as a string */

    std::string print_string;

    // if we have a checkmate evaluation
    if (abs(new_eval) > BLACK_MATED - 100) {
        bool white_mated;
        int to_mate;
        // which colour is getting mated
        if (new_eval < WHITE_MATED + 100) {
            white_mated = true;
            to_mate = -WHITE_MATED + new_eval;
        }
        else if (new_eval > BLACK_MATED - 100) {
            white_mated = false;
            to_mate = BLACK_MATED - new_eval;
        }
        else {
            throw std::runtime_error("code mistake");
        }
        // which player is getting mated (odd means them, even us)
        bool give_mate = true;
        if (to_mate % 2 == 0) {
            give_mate = false;
        }
        // adjust output based on who is getting mated
        if (give_mate) {
            to_mate = (to_mate - 1) / 2;    // moves until giving mate
            if (to_mate == 0) {
                print_string = "checkmate (win)";
            }
            else {
                print_string = "mate in " + std::to_string(to_mate) + " (win)";
            }
        }
        else {
            to_mate = to_mate / 2;          // moves until receiving mate
            if (to_mate == 0) {
                print_string = "checkmate (lose)";
            }
            else {
                print_string = "mate in " + std::to_string(to_mate) + " (lose)";
            }
        }

    }
    else {

        int num_char = 5;           // 0.000

        if (new_eval < 0) {
            num_char += 1;          // for the minus sign
        }

        print_string = std::to_string(float(new_eval) / 1000.0);
        print_string = print_string.substr(0, num_char);
    }

    return print_string;
}

void TreeEntry::print()
{
    print(true, "");
}

void TreeEntry::print(std::string starter)
{
    print(true, starter);
}

void TreeEntry::print(bool move_list_too)
{
    print(move_list_too, "");
}

void TreeEntry::print(bool move_list_too, std::string starter)
{
    int max_move_list = 5;
    
    std::cout << starter + "TreeEntry:\n";
    std::cout << starter + "\tparent_moves: ";
    for (int i = 0; i < parent_moves.size(); i++) {
        std::cout << parent_moves[i].to_letters() << "  ";
    }
    std::cout << "\n" + starter + "\tparent_keys:\n";
    for (int i = 0; i < parent_keys.size(); i++) {
        std::cout << starter + "\t\t";
        parent_keys[i].print();
    }

    std::cout << starter + "\thash_key = " << hash_key << '\n';
    std::cout << starter + "\teval = " << eval << " (" << eval / 1000.0 << ")\n";
    std::cout << starter + "\tactive = " << active << '\n';
    std::cout << starter + "\tactive_move = " << active_move << '\n';
    std::cout << starter + "\tgame_continues = " << game_continues << "\n";

    if (move_list.size() < max_move_list) {
        max_move_list = move_list.size();
    }

    if (move_list_too) {
        std::cout << starter + "\tmove_list | number of entries: "
            << move_list.size() << "  (displaying " << max_move_list << ")\n";
        for (int i = 0; i < max_move_list; i++) {
            move_list[i].print(starter + "\t\t");
        }
        std::cout << '\n';
    }
    else {
        std::cout << starter + "\tnumber of items in move_list: " 
            << move_list.size() << '\n';
    }
}

void TreeLayer::print()
{
    int max_list_print = 10;
    int list_print = 0;
    int max_board_print = 4;
    int board_print = 0;
    std::string finish;
    std::string finish_board;

    if (key_list.size() != board_list.size() or
        key_list.size() != hash_list.size()) {
        std::cout << "key list: " << key_list.size()
            << ", board_list: " << board_list.size()
            << ", hash_list: " << hash_list.size() << '\n';
        for (TreeEntry& e : board_list) {
            e.print();
        }
        throw std::runtime_error("key/board/hash list sizes are not uniform");
    }

    if (max_list_print >= key_list.size()) {
        list_print = key_list.size() - 1;
        finish = "<end>\n";
    }
    else {
        list_print = max_list_print - 1;
        finish = ".....\n";
    }

    if (max_board_print >= list_print) {
        board_print = list_print;
        finish_board = "<end>\n";
    }
    else {
        board_print = max_board_print;
        finish = ".....\n";
    }

    std::cout << "TreeLayer | number of entries: " << board_list.size() 
        << "  (displaying " << list_print + 1 << ")"
        << " | white_to_play: " << ((white_to_play) ? "true" : "false")
        << ", layer_move: " << layer_move << '\n';

    std::cout << "\thash_list | ";
    for (int i = 0; i < list_print; i++) {
        std::cout << hash_list[i] << ", ";
    }
    std::cout << hash_list[list_print] << " " + finish;

    std::cout << "\tkey_list | ";
    for (int i = 0; i < list_print; i++) {
        std::cout << key_list[i] << ", ";
    }
    std::cout << key_list[list_print] << " " + finish;

    std::cout << "\tboard_list | number of entries: " << board_list.size() 
        << "  (displaying " << board_print << ")\n";
    for (int i = 0; i < board_print + 1; i++) {
        board_list[i].print(true, "\t\t");
    }
    std::cout << "\t\t" << finish_board;
}

Lookup TreeLayer::binary_lookup(std::size_t item,
    const std::vector<std::size_t>& dictionary)
{
    /* This function finds the index for an item in a dictionary. Note for a
    dictionary of { 0 } the output index = 0, so to insert into dictionaries
    use index + 1 */

    int probe_start = 0;
    int probe_end = dictionary.size();
    int current_probe = 0;

    // create ticket to output
    Lookup receipt;
    receipt.present = false;
    receipt.index = -1;

    if (probe_end == 0) {
        receipt.index = 0;
        return receipt;
    }

    while (true) {
        // probe in the centre of the search space (int division)
        current_probe = probe_start + (probe_end - probe_start) / 2;

        //std::cout << "probe start: " << probe_start << ", probe end: "
        //    << probe_end << ", current_probe: " << current_probe << '\n';

        // have we finished?
        if (current_probe == probe_start) {
            // if the item is the same as the probe
            if (item == dictionary[current_probe]) {
                receipt.present = true;
                receipt.index = current_probe;
            }
            // new item, does it fit before the current dictionary entry
            else if (item < dictionary[current_probe]) {
                receipt.present = false;
                receipt.index = current_probe;
            }
            // or after the current dictionary entry
            else {
                receipt.present = false;
                receipt.index = current_probe + 1;
            }

            break;
        }

        // if the item is the same as the probe
        if (item == dictionary[current_probe]) {
            receipt.present = true;
            receipt.index = current_probe;
            break;
        }
        // if the item is less than the probe
        else if (item < dictionary[current_probe]) {
            probe_end = current_probe;
        }
        // if the item is greater than the probe
        else {
            probe_start = current_probe;
        }

    }

    return receipt;
}

std::vector<TreeKey> TreeLayer::remove_duplicates(std::vector<TreeKey>& old_list)
{
    /* This function removes duplicates from a TreeKey vector */

    std::vector<std::size_t> dictionary;
    std::vector<TreeKey> new_list;

    for (TreeKey& item : old_list) {

        // hash the item in the list
        std::size_t hash = treeKeyHash(item.hash());

        // find if the item is already present in the dictionary
        Lookup receipt = binary_lookup(hash, dictionary);

        // if the item is not present
        if (not receipt.present) {

            // put this non-duplicate in the new list
            new_list.push_back(item);

            // save the hash of this item in the dictionary
            dictionary.insert(dictionary.begin() + receipt.hash_insert(), hash);
        }
    }

    return new_list;
}

int TreeLayer::add(TreeEntry entry) 
{
    /* returns the location the entry has been added, or -1 if not added */

    // determine if the entry already exists
    Lookup receipt = binary_lookup(entry.hash_key, hash_list);
    // if the item is already present
    if (receipt.present) {
        return -1;
    }
    // add the entry
    board_list.push_back(entry);
    hash_list.insert(hash_list.begin() + receipt.hash_insert(), entry.hash_key);
    key_list.insert(key_list.begin() + receipt.key_insert(), board_list.size() - 1);
    return key_list[receipt.get_key()];
}

int TreeLayer::find_hash(std::size_t hash_key) 
{
    /* returns the index of the board in the layers board list, -1 if not existing */

    // determine if the hash exists in our list
    Lookup receipt = binary_lookup(hash_key, hash_list);
    // if the item is present
    if (receipt.present) {
        // what is the index in the board_list
        return key_list[receipt.get_key()];
    }
    else {
        // the item is not present
        return -1;
    }
}

NodeEval TreeLayer::find_max_eval(int entry, int sign, int active_move, int depth)
{
    /* This function returns the best evaluation at a given node, looking at 
    the move list. It does not update activity of any node! */

    NodeEval node;
    node.active = false;
    node.max_eval = (WHITE_MATED - 1) * sign; // choose a terrible eval

    if (board_list.size() <= entry) {
        std::cout << "Board_list.size() = " << board_list.size()
            << ", entry = " << entry << "\n";
        throw std::runtime_error("entry is out of bounds for board list!");
    }

    // this should indicate a checkmate or draw
    // if (board_list[entry].move_list.size() == 0) { // old code
    if (not board_list[entry].game_continues) { // new code, save if game continues

        // hence best evaluation of this node is the final board evaluation
        node.active = true;
        node.max_eval = board_list[entry].eval;

        // adjust checkmate evaluations based on how far away they are
        if (node.max_eval == WHITE_MATED or node.max_eval == BLACK_MATED) {
            node.max_eval += sign * depth;
        }

        // FOR TESTING lets confirm
        total_legal_moves_struct tlm = total_legal_moves(board_list[entry].board_state,
            (sign + 1) / 2);
        if (tlm.outcome == 0) {
            std::cout << "a board with zero move_list size has possible moves!\n";
            print_board(board_list[entry].board_state);
            throw std::runtime_error("zero moves available but not game over");
        }
        // else {
        //     print_str("Testing: confirmed checkmate board has no legal moves in find_max_eval()");
        // }
        // END TESTING
        return node;
    }

    // check that the entry is active
    if (board_list[entry].active_move != active_move) {
        //std::cout << "node is not active, it has "
        //    << board_list[entry].active_move
        //    << " whilst tree has " << active_move << '\n';
        return node;
    }

    // loop through the move list and check for the best evaluation
    for (int i = 0; i < board_list[entry].move_list.size(); i++) {
        
        // is the node active, if not, skip
        if (board_list[entry].move_list[i].active_move <= active_move) {
            //std::cout << "active move not equal, entry has "
            //    << board_list[entry].move_list[i].active_move
            //    << " whilst tree has " << active_move << '\n';

            // testing, only skip if node is deactivated
            if (board_list[entry].move_list[i].active_move < 0) {
                continue;
            }

            //continue; // comment out for testing
        }

        int next_eval = board_list[entry].move_list[i].new_eval;

        // does this evaluation beat our current best
        if (node.max_eval * sign < sign * next_eval) {
            node.max_eval = next_eval;
            //std::cout << "updated\n";
            node.active = true;
        }
    }

    return node;
}

void TreeLayer::add_finished_games(std::vector<TreeKey>& id_list)
{
    /* This function adds terminated board states to the id list to ensure that
    they are always maintained as active considerations */

    // loop through the finished games list making copies
    for (TreeKey& finished : finished_games_list) {
        id_list.push_back(finished);
    }
}

LayeredTree::LayeredTree()
{
    /* Constructor*/

    // if given no indication, set width = 5
    init(5);
    int root_outcome_ = set_root();
}

LayeredTree::LayeredTree(int width)
{
    init(width);
    int root_outcome_ = set_root();
}

LayeredTree::LayeredTree(Board board, bool white_to_play, int width)
{
    init(width);
    int root_outcome_ = set_root(board, white_to_play);
}

void LayeredTree::init(int width)
{
    width_ = width;
    current_move_ = 0;
    default_cpus_ = 1;

    // initialise with defaults
    prune_params_.base_allowance = 1000;
    prune_params_.decay = 200;
    prune_params_.min_allowance = 200;
    prune_params_.minimum_group_size = 3;
    prune_params_.use_minimum_group_size = false;
    prune_params_.max_iterations = 15;
    prune_params_.wiggle = 0.1;
}

int LayeredTree::set_root()
{
    /* Sets the root as an empty board */

    Board empty_board = create_board();
    return set_root(empty_board, true);
}

int LayeredTree::set_root(Board board, bool white_to_play)
{
    /* This function sets a root board in the tree, where future growth will
    emerge from. If the tree is empty, a first layer is created, else, the root
    is placed in the top-most layer in the tree */

    // check for checkmate or stalemate in the given board
    total_legal_moves_struct tlm = total_legal_moves(board, white_to_play);
    if (tlm.outcome != 0) {
        return tlm.outcome;
    }

    // reset variables
    new_ids_wtp_ = white_to_play;
    old_ids_wtp_ = not white_to_play;
    //new_ids_.clear();
    //old_ids_.clear();
    std::vector<TreeKey>().swap(new_ids_);
    std::vector<TreeKey>().swap(old_ids_);

    // if the tree is empty, add a layer
    if (layer_pointers_.size() == 0) {
        add_layer(1, current_move_, white_to_play);
    }

    // get the base layer of the tree
    std::shared_ptr<TreeLayer> layer_p = layer_pointers_[0];

    // prepare to add the board to the tree
    TreeEntry rootBoard(width_);
    rootBoard.board_state = board;
    rootBoard.hash_key = hash_func_(board.arr);
    rootBoard.eval = eval_board(board, white_to_play);
    rootBoard.active = true;
    rootBoard.game_continues = true;

    // add this board to the tree
    int entry_key = layer_p->add(rootBoard);

    // a node is active even with zero depth initially, before searching starts
    active_move_ = 0;

    // if the board is already present
    if (entry_key == -1) {
        entry_key = layer_p->find_hash(rootBoard.hash_key);
        // set the active move to align with the existing entry
        layer_p->board_list[entry_key].active_move = 0; // initially, no depth added to evaluation
    }
    else {
        // reset the active move
        layer_p->board_list[entry_key].active_move = 0; // initially, no depth added to evaluation
    }

    // create the key for this board
    TreeKey rootKey;
    rootKey.layer = current_move_;
    rootKey.entry = entry_key;
    rootKey.move_index = -1;
    rootKey.evaluation = rootBoard.eval;

    // add this key as the only entry in new_ids_
    new_ids_.push_back(rootKey);

    // save
    root_ = rootKey;
    root_wtp_ = white_to_play;
    root_board_ = board;

    return 0;
}

void LayeredTree::add_layer(int size, int layer_move, bool layer_wtp)
{
    /* This function adds a new layer to the tree */

    //std::shared_ptr<TreeLayer> new_layer_pointer = new TreeLayer(size);
    //layer_pointers_.push_back(new_layer_pointer);

    layer_pointers_.push_back(std::make_shared<TreeLayer>(size, width_, 
        layer_move, layer_wtp));

}

void LayeredTree::remove_layer()
{
    /* This function removes the oldest layer from the tree */

    if (layer_pointers_.size() == 0) {
        throw std::runtime_error("told to delete layer from empty tree");
    }

    //delete layer_pointers_[0];
    layer_pointers_.erase(layer_pointers_.begin());

    current_move_ += 1;
}

void LayeredTree::print()
{
    /* overload */

    print(layer_pointers_.size());
}

void LayeredTree::print(int layers)
{
    if (layers > layer_pointers_.size()) {
        layers = layer_pointers_.size();
    }

    for (int i = 0; i < layers; i++) {
        std::shared_ptr<TreeLayer> layer_p = layer_pointers_[i];
        std::cout << "LAYER " << i << ":" << '\n';
        layer_p->print();
        std::cout << "----------------------------------------\n";
    }
}

void LayeredTree::print_old_ids()
{
    std::cout << "old_ids_ | no. of entries = " << old_ids_.size() << "\n";

    if (old_ids_.size() == 0) {
        std::cout << "\t(empty)\n";
        return;
    }

    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(old_ids_[0].layer);
    Move tempMove;

    for (int i = 0; i < old_ids_.size(); i++) {
        std::cout << "\t";
        // print out all of the parent moves:
        for (int j = 0; j < layer_p->board_list[old_ids_[i].
            entry].parent_moves.size(); j++) {
            std::cout << layer_p->board_list[old_ids_[i].entry].parent_moves[j]
                .to_letters() << ", ";
        }
        old_ids_[i].print();
    }
}

void LayeredTree::print_new_ids()
{
    /* overload */

    print_new_ids(new_ids_.size());
}

void LayeredTree::print_new_ids(int max)
{
    if (max > new_ids_.size()) max = new_ids_.size();

    std::cout << "new_ids_ | no. of entries = " << new_ids_.size() << "\n";

    if (new_ids_.size() == 0) {
        std::cout << "\t(empty)\n";
        return;
    }

    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(new_ids_[0].layer);
    Move tempMove;

    for (int i = 0; i < max; i++) {
        std::cout << "\t";
        // print out all of the parent moves:
        for (int j = 0; j < layer_p->board_list[new_ids_[i].
            entry].parent_moves.size(); j++) {
            std::cout << layer_p->board_list[new_ids_[i].entry].parent_moves[j]
                .to_letters() << ", ";
        }
        new_ids_[i].print();
    }
}

void LayeredTree::print_id_list(std::vector<TreeKey> id_list)
{
    std::cout << "id_list | no. of entries = " << id_list.size() << "\n";

    if (id_list.size() == 0) {
        std::cout << "\t(empty)\n";
        return;
    }

    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(id_list[0].layer);
    Move tempMove;

    for (int i = 0; i < id_list.size(); i++) {
        std::cout << "\t";
        // print out all of the parent moves:
        for (int j = 0; j < layer_p->board_list[id_list[i].
            entry].parent_moves.size(); j++) {
            std::cout << layer_p->board_list[id_list[i].entry].parent_moves[j]
                .to_letters() << ", ";
        }
        id_list[i].print();
    }
}

std::shared_ptr<TreeLayer> LayeredTree::get_layer_pointer(int layer)
{
    /* this function returns a layer pointer, checking legal inputs */

    if (layer < current_move_) {
        std::cout << "layer is " << layer << ", current_move_ is " 
            << current_move_ << '\n';
        throw std::runtime_error("get_layer_pointer(): layer is less than current move");
    }

    int layer_index = layer - current_move_;

    if (layer_index >= layer_pointers_.size()) {
        std::cout << "layer_index is " << layer_index << ", size() is " 
            << layer_pointers_.size() << '\n';
        throw std::runtime_error("layer_index is greater than layer_pointer_ length");
    }

    return layer_pointers_[layer_index];
}

std::shared_ptr<TreeLayer> LayeredTree::get_or_create_layer_pointer(int layer)
{
    /* this function gets an existing layer pointer, or creates a new layer in the tree */

    if (layer < current_move_) {
        std::cout << "layer is " << layer << ", current_move_ is " 
            << current_move_ << '\n';
        throw std::runtime_error("get_or_create_layer_pointer(): layer is less than current move");
    }

    int layer_index = layer - current_move_;

    // check if the layer index exceeds the size of the tree by more than one
    if (layer_index > layer_pointers_.size()) {
        std::cout << "layer_index is " << layer_index << ", size() is " 
            << layer_pointers_.size() << '\n';
        throw std::runtime_error("layer_index is greater than one more than layer_pointer_ length");
    }
    else if (layer_index == layer_pointers_.size()) {
        // create a new layer pointer for this new layer
        std::shared_ptr<TreeLayer> prev_layer_p = get_layer_pointer(layer - 1);
        int last_layer_move = prev_layer_p->layer_move;
        bool last_layer_wtp = prev_layer_p->white_to_play;
        add_layer(width_, last_layer_move + 1, not last_layer_wtp);
    }

    return layer_pointers_[layer_index];
}

TreeKey LayeredTree::add_move(TreeKey parent_key, move_struct& move)
{
    /* This function adds a move to the layer as well as adding it to the
    parent in the previous layer */

    TreeKey new_key;
    new_key.layer = parent_key.layer + 1;
    new_key.entry = -1;    // defaults to indicate it is not set
    new_key.move_index = -1;
    new_key.evaluation = move.evaluation;

    // get the pointers to the current layer and previous layers
    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(parent_key.layer + 1);
    std::shared_ptr<TreeLayer> prev_layer_p = get_layer_pointer(parent_key.layer);

    // extract the hash of the resultant board
    std::size_t board_hash = hash_func_(move.board.arr);

    // input information about the move
    MoveEntry moveEntry;
    moveEntry.move.start_sq = move.start_sq;
    moveEntry.move.dest_sq = move.dest_sq;
    moveEntry.move.move_mod = move.move_mod;
    moveEntry.new_eval = move.evaluation;
    moveEntry.new_hash = board_hash;

    moveEntry.active_move = 0; // add with evaluation to 0 depth

    // find out if this position is already in the current layer
    int key = layer_p->find_hash(board_hash);

    //std::cout << "the output of find_hash was " << key << '\n';

    // if this board state already exists
    if (key != -1) {

        // check to see if this parent already exists
        for (const TreeKey& parent : layer_p->board_list[key].parent_keys) {
            if (parent.layer == parent_key.layer and
                parent.entry == parent_key.entry) {
                // nothing to add, this entry is already done
                new_key.entry = key;
                new_key.evaluation = layer_p->board_list[key].eval;
                return new_key;
            }
        }

        // add the move to the parent in the prev. layer, and save the move index
        prev_layer_p->board_list[parent_key.entry].move_list.push_back(moveEntry);
        parent_key.move_index = prev_layer_p->board_list[parent_key.entry]
            .move_list.size() - 1;

        // this parent is new to this board state, hence add it
        layer_p->board_list[key].parent_keys.push_back(parent_key);

        new_key.entry = key;
        new_key.evaluation = layer_p->board_list[key].eval;


        return new_key;
    }

    // add the move to the board list of the parent, store its position in parent key
    if (prev_layer_p->board_list.size() <= parent_key.entry) {
        throw std::runtime_error("board list smaller than parent key.entry");
    }
    prev_layer_p->board_list[parent_key.entry].move_list.push_back(moveEntry);
    parent_key.move_index = prev_layer_p->board_list[parent_key.entry]
        .move_list.size() - 1;

    // since this board state does not exist, create new tree entry
    TreeEntry treeEntry(width_);
    treeEntry.parent_keys.push_back(parent_key);
    treeEntry.board_state = move.board;
    treeEntry.hash_key = board_hash;
    treeEntry.eval = move.evaluation;
    treeEntry.active = true;
    treeEntry.game_continues = true;

    // track the parent moves - not these are NOT unique, only a useful guide
    treeEntry.parent_moves = prev_layer_p->board_list[parent_key.entry].parent_moves;
    treeEntry.parent_moves.push_back(moveEntry.move);

    // the new layer is not included in cascade, so +1
    treeEntry.active_move = 0; // add with evaluation to zero depth

    // add this new entry to the layer
    int generated_key = layer_p->add(treeEntry);

    if (generated_key == -1) {
        throw std::runtime_error("adding treeEntry failed - already exists!");
    }

    new_key.entry = generated_key;
    new_key.evaluation = move.evaluation;


    return new_key;
}

std::vector<TreeKey> LayeredTree::add_board_replies(const TreeKey& parent_key,
    generated_moves_struct& gen_moves)
{
    /* This function adds the generated moves to the tree in the next layer, as
    well as updating the parent in the base layer. It returns the keys of the
    replies in a vector, ordered from best to worst */

    if (parent_key.layer < current_move_) {
        throw std::runtime_error("add_board_replies(): layer is less than current move");
    }

    // check if the board terminates (checkmate/draw)
    if (gen_moves.game_continues == false) {

        // mark the parent state as having a finished game
        std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(parent_key.layer);
        layer_p->board_list[parent_key.entry].game_continues = false;

        // create an key entry for this terminated board state
        TreeKey finishedGame;
        finishedGame.layer = parent_key.layer;
        finishedGame.entry = parent_key.entry;
        finishedGame.move_index = -2;
        finishedGame.evaluation = gen_moves.base_evaluation;

        // save the key of this board state in its corresponding layer
        layer_p->finished_games_list.push_back(finishedGame);

        // return an empty vector, there are no replies
        std::vector<TreeKey> empty_vector;
        return empty_vector;
    }

    // how many moves are available
    int num_loops = 0;
    int width = width_;
    if (gen_moves.moves.size() > width) {
        num_loops = width;
    }
    else {
        num_loops = gen_moves.moves.size(); 
    }

    // create a vector to output all of the new keys, ordered best to worst
    std::vector<TreeKey> reply_keys(num_loops);

    // if the board has zero legal moves
    if (num_loops == 0) {
        throw std::runtime_error("trying to add moves to board with no legal moves");
    }

    // loop through the best generated moves to set width
    bool print_evals = false; // for testing
    if (print_evals) std::cout << "Generated move evals, white to play = " << gen_moves.white_to_play << ": ";
    for (int i = 0; i < num_loops; i++) {

        // add the moves to the tree
        TreeKey new_key = add_move(parent_key, gen_moves.moves[i]);
        new_ids_.push_back(new_key);

        added_ids_.push_back(new_key);

        reply_keys[i] = new_key;

        if (print_evals) std::cout << new_key.evaluation << " (layer = " << new_key.layer << ")" << "; ";
    }
    if (print_evals) std::cout << "\n";

    // for cutoff prune
    new_ids_groups_.push_back(num_loops);
    new_ids_parents_.push_back(parent_key);

    return reply_keys;
}

bool LayeredTree::grow_tree()
{
    /* This function loops through all items in the old_ids_ vector and 
    generates replies to those boards. These replies are then added to
    the next layer of the tree */

    if (old_ids_.size() == 0) {
        throw std::runtime_error("old_ids_ is empty!");
    }

    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(old_ids_[0].layer);

    // what was white to play in the previous layer
    bool prev_to_play = layer_p->white_to_play;

    // do we need to add a layer to the tree
    int layer_index = old_ids_[0].layer - current_move_;
    if (layer_index + 1 == layer_pointers_.size()) {
        add_layer(old_ids_.size() * (width_ + 1), active_move_, not prev_to_play);
    }

    bool white_to_play = old_ids_wtp_;

    // loop through the old_ids_ and make new moves
    for (const TreeKey& id : old_ids_) {

        // what is our board for this id
        Board board = layer_p->board_list[id.entry].board_state;

        // generate responses on this board, then add these to the tree
        generated_moves_struct gen_moves = generate_moves(board, white_to_play);
        add_board_replies(id, gen_moves);

        boards_checked_ += 1;
    }

    // if we were unable to grow the tree
    if (new_ids_.size() == 0) {
        return false;
    }

    return true;
}

void generate_thread(std::vector<ThreadOut>& output, bool white_to_play)
{
    /* generate board moves in a thread */

    for (ThreadOut& x : output) {
        // generate responses on this board
        x.gen_moves = generate_moves(x.board, white_to_play);
    }
}

bool LayeredTree::grow_tree_threaded(int num_cpus)
{
    /* grow the tree in multiple threads at once */

    // if only one cpu, use the original grow_tree() function
    if (num_cpus == 1) {
        return grow_tree();
    }

    int num_ids = old_ids_.size();

    // if not enough ids to make threading worthwhile
    if (num_ids < 100) {
        return grow_tree();
    }

    print_str("THREADING NOW");

    if (old_ids_.size() == 0) {
        throw std::runtime_error("old_ids_ is empty!");
    }

    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(old_ids_[0].layer);

    // what was white to play in the previous layer
    bool prev_to_play = layer_p->white_to_play;

    // do we need to add a layer to the tree
    int layer_index = old_ids_[0].layer - current_move_;
    if (layer_index + 1 == layer_pointers_.size()) {
        add_layer(old_ids_.size() * (width_ + 1), active_move_, not prev_to_play);
    }

    bool white_to_play = old_ids_wtp_;

    // prepare for threading
    int num_threads = num_cpus - 1;
    int ids_per_thread = num_ids / num_cpus;
    int ids_final_thread = num_ids - (num_cpus - 1) * ids_per_thread;

    // create a vector of pointers
    std::vector<std::shared_ptr<std::vector<ThreadOut>>> thread_out_ptrs;

    // loop through and prepare threads
    for (int c = 0; c < num_cpus; c++) {

        // create a new pointer to a vector of thread objects
        thread_out_ptrs.push_back(std::make_shared<std::vector<ThreadOut>>());

        // how many ids are going to this thread
        int loops;
        if (c != num_cpus - 1) {
            loops = ids_per_thread;
        }
        else {
            loops = ids_final_thread;
        }

        // fill up the thread vector with ids
        for (int j = 0; j < loops; j++) {
            // make the thread storage object, fill with id and board, add
            ThreadOut tout;
            tout.id = old_ids_[j + c * ids_per_thread];
            tout.board = layer_p->board_list
                [old_ids_[j + c * ids_per_thread].entry].board_state;
            thread_out_ptrs[c]->push_back(tout);
        }
    }

    // launch the extra threads
    std::vector<std::thread> threads(num_threads);
    for (int t = 0; t < num_threads; t++) {
        threads[t] = std::thread(generate_thread, std::ref(*thread_out_ptrs[t]), 
            white_to_play);
    }

    // compute in the main thread
    generate_thread(*thread_out_ptrs[num_threads], white_to_play);

    // wait for each thread to complete
    for (int t = 0; t < num_threads; t++) {
        threads[t].join();
    }

    // now build up our tree using the output of the threads
    for (int t = 0; t < num_cpus; t++) {
        for (ThreadOut& x : *thread_out_ptrs[t]) {
            add_board_replies(x.id, x.gen_moves);
            boards_checked_ += 1;
        }
    }

    // if we were unable to grow the tree
    if (new_ids_.size() == 0) {
        return false;
    }

    return true;
}

bool LayeredTree::check_game_continues()
{
    /* This function checks whether the board can continue to be searched */

    if (root_outcome_ != 0) {
        return false;
    }
    if (new_ids_.size() == 0) {
        return false;
    }

    return true;
}

void LayeredTree::advance_ids()
{
    /* This function turns the new_ids_ into old_ids_, and clears the new_ids_
    vector to be ready for another step. Duplicates should also be removed */

    // move new ids into old_ids_ vector, then clear
    old_ids_ = new_ids_; // remove duplicates in this step later
    //new_ids_.clear();

    std::vector<TreeKey>().swap(new_ids_);

    // swap the booleans for whoevers turn is next
    old_ids_wtp_ = new_ids_wtp_;
    new_ids_wtp_ = not new_ids_wtp_;

    // for cutoff prune
    //new_ids_groups_.clear();
    //new_ids_parents_.clear();
    std::vector<int>().swap(new_ids_groups_);
    std::vector<TreeKey>().swap(new_ids_parents_);
}

void LayeredTree::cascade()
{
    /* This function cascades from the bottom to the top of the tree, updating
    evaluations and node activity */

    std::vector<TreeKey> id_list = old_ids_;
    std::vector<TreeKey> next_id_list;

    std::shared_ptr<TreeLayer> this_layer_p;
    std::shared_ptr<TreeLayer> next_layer_p;

    // we cascade through all layers except the very bottom, which is unfinished
    int depth = layer_pointers_.size() - 2;

    // loop backwards through the layers
    for (int k = 0; k <= depth; k++) {

        if (id_list.size() == 0) {
            throw std::runtime_error("empty id_list!");
        }

        int layer = depth - k + current_move_;
        this_layer_p = get_layer_pointer(layer);

        int sign = 2 * this_layer_p->white_to_play - 1;

        int offset = depth - k;              // distance from top of tree

        if (offset != 0) {
            // get the pointer to the next/previous layer
            next_layer_p = get_layer_pointer(layer - 1);
        }

        // add checkmate and drawn game nodes to the list
        this_layer_p->add_finished_games(id_list);

        // remove duplicates from the list
        id_list = this_layer_p->remove_duplicates(id_list);

        // loop through the id items
        for (TreeKey& id_item : id_list) {

            // find the best evaluation at this node
            NodeEval node = this_layer_p->find_max_eval(id_item.entry, sign, 
                active_move_, offset);

            if (not node.active) {
                //// TESTING
                //// set the parents move not active as well, override shared parents
                //for (const TreeKey& parent : this_layer_p->
                //    board_list[id_item.entry].parent_keys) {
                //    
                //    next_layer_p->board_list[parent.entry].move_list[parent.move_index]
                //        .active_move = -8;
                //}
                //// END TESTING
                continue;
            }

            // now we have the best evaluation at this node

            // overwrite the evaluation of this node
            this_layer_p->board_list[id_item.entry].eval = node.max_eval;

            // keep the node active
            this_layer_p->board_list[id_item.entry].active_move += 1;

            // end here if on the final loop
            if (offset == 0) {
                if (id_list.size() != 1) {
                    throw std::runtime_error("id_list not == 1 on final cascade");
                }
                break;
            }

            // update this evaluation and activity in each parent
            for (const TreeKey& parent : this_layer_p->
                    board_list[id_item.entry].parent_keys) {

                next_layer_p->board_list[parent.entry].move_list[parent.move_index]
                    .new_eval = node.max_eval;

                // update parent activity in move list
                next_layer_p->board_list[parent.entry].move_list[parent.move_index]
                    .active_move += 1;

                // add the parent to the next id list
                next_id_list.push_back(parent);
            }      
        }

        // now we have finished with this id list
        id_list = next_id_list;
        //next_id_list.clear();
        std::vector<TreeKey>().swap(next_id_list); // swap with empty vector

        // loop to next layer up
    }

    // now we have finished, increment the active move
    active_move_ += 1;
}

std::vector<TreeKey> LayeredTree::my_sort(std::vector<TreeKey>& vec)
{
    /* Very basic sorting algorithm */

    std::vector<TreeKey> out;

    if (vec.size() == 0) {
        throw std::runtime_error("vector for sorting has length zero");
    }

    // put the first item into output
    out.push_back(vec[0]);

    //for (const TreeKey& item : vec) {
    for (int i = 1; i < vec.size(); i++) {

        TreeKey& item = vec[i];

        int probe;
        int probe_start = 0;
        int probe_end = out.size();

        while (true) {

            probe = probe_start + (probe_end - probe_start) / 2;

            // are we finished
            if (probe == probe_start or item.evaluation == out[probe].evaluation) {
                // insert into the list
                if (item.evaluation <= out[probe].evaluation) {
                    out.insert(out.begin() + probe, item);
                }
                else {
                    out.insert(out.begin() + probe + 1, item);
                }
                break;
            }

            if (item.evaluation < out[probe].evaluation) {
                probe_end = probe;
            }
            else {
                probe_start = probe;
            }
        }
    }

    return out;
}

std::vector<TreeKey> LayeredTree::remove_duplicates(std::vector<TreeKey>& vec)
{
    /* function to remove duplicates from a vector of tree keys */

    std::shared_ptr<TreeLayer> this_layer_p = get_layer_pointer(current_move_);

    return this_layer_p->remove_duplicates(vec);
}

void LayeredTree::mark_for_deactivation(std::vector<TreeKey> vec, bool deactivate)
{
    /* mark the node for deactivation if deactivate = true, otherwise, mark the
    node as not possible to deactivate */

    if (vec.size() == 0) {
        return;
    }

    std::shared_ptr<TreeLayer> base_layer_ptr = layer_pointers_[0];

    // make pointers so we can swap between deactivate and reactivate
    std::vector<std::size_t>* dictionary;
    std::vector<TreeKey>* node_list;

    // are we adding to the deactivate list, or the reactivate list
    if (deactivate) {
        dictionary = &deactivation_hash;
        node_list = &deactivation_nodes;
    }
    else {
        dictionary = &reactivation_hash;
        node_list = NULL; // not needed
    }

    for (TreeKey& t : vec) {

        std::size_t t_hash = base_layer_ptr->treeKeyHash(t.hash());
        Lookup receipt = base_layer_ptr->binary_lookup(t_hash, *dictionary);

        // if this node is not already in the list
        if (not receipt.present) {
            
            if (deactivate) {
                // add the node to the list to be deactivated
                node_list->push_back(t);
            }
            
            // add the hash to our dictionary so we know this node is added
            dictionary->insert(dictionary->begin() + receipt.index, t_hash);
        }
    }
}

void LayeredTree::deactivate_nodes()
{
    /* deactivate the nodes, so long as they aren't in the reactivate list */

    std::shared_ptr<TreeLayer> base_layer_ptr = layer_pointers_[0];

    int test_counter = 0;

    for (TreeKey& t : deactivation_nodes) {

        // see if this key exists in the reactivation list
        std::size_t t_hash = base_layer_ptr->treeKeyHash(t.hash());
        Lookup receipt = base_layer_ptr->binary_lookup(t_hash, reactivation_hash);

        // if it doesn't, deactivate this node
        if (not receipt.present) {
            deactivate_node(t);
            test_counter += 1;
        }
    }

    std::cout << "The number of nodes deactivated: " << test_counter
        << ", the number of nodes in the deactivate list "
        << deactivation_nodes.size() << ", the number kept active "
        << reactivation_hash.size() << '\n';

    // now we are finished, clear everything
    deactivation_hash.clear();
    deactivation_nodes.clear();
    reactivation_hash.clear();
}

void LayeredTree::deactivate_node(TreeKey& node)
{
    /* deactivate an individual node */

    std::shared_ptr<TreeLayer> this_layer_p = get_layer_pointer(node.layer);
    std::shared_ptr<TreeLayer> prev_layer_p = get_layer_pointer(node.layer - 1);

    // deactivate this node
    this_layer_p->board_list[node.entry].active_move = -3;

    // loop through node parents, to deactive move child in move list
    for (int c = 0; c < this_layer_p->board_list[node.entry]
        .parent_keys.size(); c++) {

        // look up the parent key and then deactivate child in move list
        prev_layer_p->board_list
            [this_layer_p->board_list[node.entry]
            .parent_keys[c].entry]
        .move_list
            [this_layer_p->board_list[node.entry]
            .parent_keys[c].move_index]
        .active_move = -4;
    }
}

void LayeredTree::limit_prune()
{
    /* This function prunes the new_ids_ list using a simple cut-off */

    std::vector<TreeKey> ids_pruned;

    std::shared_ptr<TreeLayer> this_layer_p = get_layer_pointer(new_ids_[0].layer);
    std::shared_ptr<TreeLayer> prev_layer_p = get_layer_pointer(new_ids_[0].layer - 1);

    // remove duplicates here, they are boards that share two parents
    std::vector<TreeKey> ids_unique = this_layer_p->remove_duplicates(new_ids_);

    // sort the list
    std::sort(ids_unique.begin(), ids_unique.end());
    std::vector<TreeKey> ids_copy = ids_unique;
    //std::vector<TreeKey> ids_copy = my_sort(ids_unique);

    // implement a basic restriction
    int limit = 10000;
    //int limit = ids_copy.size() - 2;

    int direction = 0;
    int start = 0;
    if (not new_ids_wtp_) {
        direction = 1;
        start = 0;
    }
    else {
        direction = -1;
        start = ids_copy.size() - 1;
    }

    for (int i = 0; i < ids_copy.size(); i++) {

        int j = start + (i * direction);
        // save the best new_ids
        ids_pruned.push_back(ids_copy[j]);

        // once we hit the pruning limit, admit no more ids
        if (i == limit - 1) {

            std::cout << "the best evaluation (i=0) is: ";
            ids_pruned[0].print();
            std::cout << "the worst evalution (i=" << i << ") is: ";
            ids_pruned[i].print();

            // now loop through the remaining node to deactivate them
            for (int a = i; a < ids_copy.size(); a++) {
                int b = start + (a * direction);
                deactivate_node(ids_copy[b]);

                /* old code, now wrapped into a function
                // deactivate the node
                this_layer_p->board_list[ids_copy[b].entry].active_move = -3;

                // loop through node parents, to deactive move child in move list
                for (int c = 0; c < this_layer_p->board_list[ids_copy[b].entry]
                    .parent_keys.size(); c++) {
                    // look up the parent key and then deactivate child in move list
                    prev_layer_p->board_list
                        [this_layer_p->board_list[ids_copy[b].entry]
                            .parent_keys[c].entry]
                        .move_list
                        [this_layer_p->board_list[ids_copy[b].entry]
                            .parent_keys[c].move_index]
                        .active_move = -4;
                }
                //std::cout << "Deactivated a node" << '\n';
                */
            }
            // break as we reached our limit
            break;
        }
    }

    //std::cout << "new_ids_wtp is " << new_ids_wtp_ << '\n';
    //std::cout << "before sorting\n";
    //print_new_ids();

    new_ids_ = ids_pruned;

    //std::cout << "after sorting (and pruning)\n";
    //print_new_ids();

}

void LayeredTree::recursive_prune(std::vector<TreeKey>& id_set, TreeKey parent,
    int k, int kmax, bool deactivate)
{
    /* recursive pruning function */

    if (id_set.size() == 0) return;
    if (k == kmax) return;  // safety check, should never be triggered

    std::shared_ptr<TreeLayer> this_layer_p;
    this_layer_p = get_layer_pointer(parent.layer);
    int sign = 2 * this_layer_p->white_to_play - 1;

    int cutoff;

    // for testing
    bool test_mode = false;
    if (parent.layer == root_.layer + 1) {
        test_mode = true;
    }
    
    // define the cutoff at this layer
    if (k == 0) {
        cutoff = id_set[0].evaluation * sign - prune_params_.base_allowance;
    }
    else {
        // get the best evaluation at the parent to set the cutoff
        int max_eval = this_layer_p->board_list[parent.entry].eval;
        int allowance = prune_params_.base_allowance - k * prune_params_.decay;
        if (allowance < prune_params_.min_allowance)
            allowance = prune_params_.min_allowance;
        cutoff = max_eval * sign - allowance;

        //if (test_mode) {
        //    std::cout << "Max eval is " << max_eval << ", cutoff is "
        //        << cutoff << ", and sign " << sign
        //        << ", the parent entry is:\n";
        //    parent.print();
        //}
    }

    // test a very robust method
    std::vector<TreeKey> new_set;
    std::vector<TreeKey> deactivate_set;

    // loop backwards through the id_set
    for (int i = 0; i < id_set.size(); i++) {

        //if (test_mode) {
        //    std::cout << "\tid number " << i << " has evaluation "
        //        << id_set[i].evaluation << " which has ";
        //}

        // check if the ids have passed the cuttoff
        if (id_set[i].evaluation * sign < cutoff) {
            deactivate_set.push_back(id_set[i]);     

            //if (test_mode) std::cout << "been pruned\n";

        }
        else {
            new_set.push_back(id_set[i]);

            //if (test_mode) std::cout << "passed\n";
        }
    }

    // testing super robust method
    if (deactivate and deactivate_set.size() != 0) {
        mark_for_deactivation(deactivate_set, true);
    }

    // keep only the ids that passed
    id_set = new_set;

    // check that we do not fall below the minimum group size
    if (prune_params_.use_minimum_group_size) {
        if (id_set.size() < prune_params_.minimum_group_size) {
            id_set.clear();
        }
    }

    // if all ids have been pruned
    if (id_set.size() == 0) {
        return;
    }

    // we stop short of the last layer, there are no more parents to prune
    if (k == kmax - 1) return;

    // if there is only one parent
    if (this_layer_p->board_list[parent.entry].parent_keys.size() == 1) {
        recursive_prune(id_set, this_layer_p->board_list[parent.entry].parent_keys[0],
            k + 1, kmax, deactivate);
    }
    else {
        // make a copy of the id set
        std::vector<TreeKey> id_copy = id_set;

        // prune using the first parent
        recursive_prune(id_copy, this_layer_p->board_list[parent.entry].parent_keys[0],
            k + 1, kmax, deactivate);

        for (int p = 1; p < this_layer_p->board_list[parent.entry].
            parent_keys.size();  p++) {
            // make a new copy of the id set
            std::vector<TreeKey> id_copy_new = id_set;
            // prune
            recursive_prune(id_copy_new, 
                this_layer_p->board_list[parent.entry].parent_keys[p], k + 1, kmax,
                deactivate);
            // keep the least pruned set, just in case
            if (id_copy_new.size() > id_copy.size()) {
                id_copy = id_copy_new;
            }
        }

        // finally, use the least pruned set that we saved
        id_set = id_copy;
    }
}

std::vector<TreeKey> LayeredTree::cutoff_prune()
{
    /* overload */
    bool deactivate = true;
    return cutoff_prune(deactivate);
}

std::vector<TreeKey> LayeredTree::cutoff_prune(bool deactivate)
{
    /* prune the new_ids_ using an evaluation cutoff */

    if (new_ids_.size() == 0) {
        throw std::runtime_error("cutoff_prune has been given no new_ids_");
    }

    int depth = new_ids_[0].layer - current_move_;

    // set up parameters
    int old_eval = 0;
    int kmin = 0;
    int kmax = depth;
    int lower_bound = 0;
    int i = 0;

    std::vector<TreeKey> ids_pruned;
    std::vector<TreeKey> starting_set;
    std::vector<TreeKey> new_set;

    bool cascade_player = old_ids_wtp_;

    //std::cout << "about to start pruning:\n";
    //print_vector(new_ids_groups_, "id groups");
    //print_vector(new_ids_parents_, "id parents");

    //for (int n : new_ids_groups_) {
    for (int i = 0; i < new_ids_groups_.size(); i++) {

        int n = new_ids_groups_[i];

        if (n == 0) continue;

        // build the next input set for pruning
        for (int j = lower_bound; j < lower_bound + n; j++) {
            starting_set.push_back(new_ids_[j]);
            //std::cout << "j is " << j << '\n';
        }

        //print_vector(starting_set, "starting set before pruning");

        // extract the parent of this set
        TreeKey parent = new_ids_parents_[i];

        // prune and return the pruned version of the starting set
        recursive_prune(starting_set, parent, kmin, kmax, deactivate);

        //print_vector(starting_set, "starting set after pruning");

        // add these to the pruned set
        for (TreeKey& t : starting_set) {
            ids_pruned.push_back(t);
        }

        //i += 1; // counter
        starting_set.clear();
        lower_bound += n;
    }

    // remove any duplicates in the set
    ids_pruned = remove_duplicates(ids_pruned);

    if (deactivate) {
        // mark our final set as not possible to deactivate
        mark_for_deactivation(ids_pruned, false);

        // deactivate all the pruned nodes
        deactivate_nodes();
    }

    return ids_pruned;
}

void LayeredTree::target_prune(int target = 0)
{
    /* Cutoff prune but repeated to aim for a particular target */

    // for testing
    bool debug = true;

    std::vector<TreeKey> pruned_ids;
    int base_allowance = prune_params_.base_allowance;
    int min_allowance = prune_params_.min_allowance;
    int decay = prune_params_.decay;
    int max_iters = prune_params_.max_iterations;
    float wiggle = prune_params_.wiggle;
    bool deactivate;

    if (target < 1) {
        // if no target is set, prune only once with standard settings
        max_iters = 0;
    }
    if (new_ids_.size() < target) {
        // if we have less items than the target, no pruning needed
        return;
    }

    if (debug)
        std::cout << "Target prune starts with " << new_ids_.size() << " new_ids\n";

    for (int i = 0; i < max_iters; i++) {

        deactivate = false;
        pruned_ids.clear();
        pruned_ids = cutoff_prune(deactivate);

        float ratio = (float(target) / pruned_ids.size());

        if (debug) {
            std::cout << "Pruning step " << i + 1 << " gives "
                << pruned_ids.size() << " ids (target " << target << ", ratio "
                << ratio << ")";
            std::cout << " | params: (base/decay/min): "
                << prune_params_.base_allowance << " / "
                << prune_params_.decay << " / "
                << prune_params_.min_allowance << "\n";
        }

        // if we fall within our desired wiggle room
        if (ratio > 1 - wiggle and ratio < 1 + wiggle) {
            break;
        }

        // set a max ratio to damp oscillation
        if (ratio > 1) ratio = 1.1;
        if (ratio < 1) ratio = 0.9;

        //if (ratio > 1) break;
        
        prune_params_.base_allowance = static_cast<int>(prune_params_.base_allowance
            * ratio);
        prune_params_.min_allowance = static_cast<int>(prune_params_.min_allowance
            * ratio);
        prune_params_.decay = static_cast<int>(prune_params_.decay * ratio);
    }

    // now we want to deactivate nodes, repeat the last prune
    deactivate = true;
    new_ids_ = cutoff_prune(deactivate);
    
    //std::cout << "Pruning finished, these are the best moves:\n";
    //print_best_moves(true);
    //print_vector(new_ids_, "new_ids_");

    // restore pruning parameters to defaults
    prune_params_.base_allowance = base_allowance;
    prune_params_.min_allowance = min_allowance;
    prune_params_.decay = decay;
}

std::vector<MoveEntry> LayeredTree::get_dead_moves(TreeKey node)
{
    /* Get the dead move nodes at the root, for printing */

    // get the move list at the root position
    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(node.layer);
    std::vector<MoveEntry> copy_moves = layer_p->board_list[node.entry].move_list;
    std::sort(copy_moves.begin(), copy_moves.end());

    std::vector<MoveEntry> output_moves;
    bool white_to_play = layer_p->white_to_play;

    // do we loop forwards or backwards depending on white to play
    int direction = 0;
    int start = 0;
    if (not white_to_play) {
        direction = 1;
        start = 0;
    }
    else {
        direction = -1;
        start = copy_moves.size() - 1;
    }

    int printed = 0;
    //int limit = width_;
    int limit = copy_moves.size();

    for (int i = 0; i < copy_moves.size(); i++) {
        // index
        int j = start + (i * direction);
        // if the move is active, ignore it, we want dead moves only
        if (copy_moves[j].active_move < active_move_) {
            output_moves.push_back(copy_moves[j]);
            printed += 1;
            if (printed == limit) {
                break;
            }
        }
    }

    return output_moves;
}

std::vector<MoveEntry> LayeredTree::get_best_moves()
{
    /* get the best moves from the root */

    return get_best_moves(root_);
}

std::vector<MoveEntry> LayeredTree::get_best_moves(TreeKey node)
{
    /* This function returns the best moves at a given node */

    // get the move list at the root position
    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(node.layer);
    std::vector<MoveEntry> copy_moves = layer_p->board_list[node.entry].move_list;
    std::sort(copy_moves.begin(), copy_moves.end());

    std::vector<MoveEntry> output_moves;
    bool white_to_play = layer_p->white_to_play;

    // do we loop forwards or backwards depending on white to play
    int direction = 0;
    int start = 0;
    if (not white_to_play) {
        direction = 1;
        start = 0;
    }
    else {
        direction = -1;
        start = copy_moves.size() - 1;
    }

    int printed = 0;
    //int limit = width_;
    int limit = copy_moves.size();

    for (int i = 0; i < copy_moves.size(); i++) {
        // index
        int j = start + (i * direction);
        // new comment: fine if evaluated to or deeper than the 'active_move_' depth
        if (copy_moves[j].active_move >= active_move_) {
            output_moves.push_back(copy_moves[j]);
            printed += 1;
            if (printed == limit) {
                break;
            }
        }
    }

    return output_moves;
}

void LayeredTree::print_best_move()
{
    /* Print the best move found in the tree at the root */

    // first get the best moves at this node
    std::vector<MoveEntry> best_moves = get_best_moves(root_);

    std::string to_print = "The computer calculated the strongest next move as "
        + best_moves[0].print_move() + " with evaluation " + best_moves[0].print_eval();
    print_str(to_print);
}

void LayeredTree::print_best_moves()
{
    /* overload */

    print_best_moves(root_, false);
}

void LayeredTree::print_best_moves(TreeKey node)
{
    /* overload */

    print_best_moves(node, false);
}

void LayeredTree::print_best_moves(bool dead_nodes)
{
    /* overload, print at the root */

    print_best_moves(root_, dead_nodes);
}

void LayeredTree::print_best_moves(TreeKey node, bool dead_nodes)
{
    /* This function prints the move considerations to the console */

    // first get the best moves at this node
    std::vector<MoveEntry> best_moves = get_best_moves(node);

    print_str("The best moves are:");
    for (int i = 0; i < best_moves.size(); i++) {
        std::string to_print = "\t" + best_moves[i].print_move() + " with eval " 
            + best_moves[i].print_eval() + "\t(Evaluated to depth "
            + std::to_string(best_moves[i].active_move) + ")";
        print_str(to_print);
    }

    // print the dead nodes too, if indicated
    if (dead_nodes) {
        std::vector<MoveEntry> dead_moves = get_dead_moves(node);
        for (int i = 0; i < dead_moves.size(); i++) {
            std::string to_print = "\t" + dead_moves[i].print_move() + " with eval " 
                + dead_moves[i].print_eval() +  + "\t(Ignored, evaluated to depth "
            + std::to_string(dead_moves[i].active_move) + ")";
            print_str(to_print);
        }
    }
}

void LayeredTree::print_boards_checked()
{
    print_str("Boards checked: " + std::to_string(boards_checked_));
}

void LayeredTree::print_node_board(TreeKey node)
{
    /* print the board at the root of a node */

    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(node.layer);
    print_board(layer_p->board_list[node.entry].board_state);
}

bool LayeredTree::next_layer(int layer_width, int num_cpus)
{
    /* overload */

    width_ = layer_width;
    default_cpus_ = num_cpus;

    //prune_params_.minimum_group_size = width_ / 2;

    return next_layer();
}

bool LayeredTree::next_layer()
{
    /* This function advances the tree by one layer, but does no pruning */

    // check the game continues
    if (not check_game_continues()) return false;

    advance_ids();
    grow_tree_threaded(default_cpus_);
    cascade();

    // check the game continues
    if (not check_game_continues()) return false;

    return true;
}

bool LayeredTree::test_dictionary()
{
    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(1);

    bool passed = true;

    std::cout << "dictionary: ";

    for (int i = 1; i < layer_p->hash_list.size() - 1; i++) {
        if (layer_p->hash_list[i] > layer_p->hash_list[i + 1]) {
            passed = false;
        }
        std::cout << layer_p->hash_list[i] << ", ";
    }
    std::cout << layer_p->hash_list[layer_p->hash_list.size() - 1] << ".";
    std::cout << '\n';
    return passed;
}

std::vector<TreeKey> LayeredTree::add_depth_at_key(TreeKey key)
{
    /* generate a set of move replies at a given key (board state), and then add them into the tree */

    // get layer pointer for this key and the next layer (for the replies)
    std::shared_ptr<TreeLayer> key_layer_p = get_layer_pointer(key.layer);
    std::shared_ptr<TreeLayer> next_layer_p = get_or_create_layer_pointer(key.layer + 1); // may grow the tree

    // determine the board and who plays next
    Board board = key_layer_p->board_list[key.entry].board_state;
    bool white_to_play = key_layer_p->white_to_play;

    // now generate and evaluate all possible replies on this board
    generated_moves_struct gen_moves;
    if (use_nn_eval) {
        #if defined(LUKE_PYTORCH)
            gen_moves = generate_moves_nn(board, white_to_play);
        #else
            throw std::runtime_error("LUKE_PYTORCH not defined, but LayeredTree::use_nn_eval = True");
        #endif
    }
    else {
        gen_moves = generate_moves(board, white_to_play);
    }

    // std::cout << "After generate moves\n" << std::flush;

    // add replies into next layer of tree (up to limit = width_)
    std::vector<TreeKey> reply_keys = add_board_replies(key, gen_moves);
    boards_checked_ += 1;

    evaluated_ids_.push_back(key);

    return reply_keys;
}

int LayeredTree::update_upstream_evaluations()
{
    /* overload allowing function to be called without inputs */

    // inputs any new board positions which have been evaluated but not updated
    int best_eval = update_upstream_evaluations(evaluated_ids_);

    // wipe the above vector, which have now been updated
    std::vector<TreeKey>().swap(evaluated_ids_);

    return best_eval;
}

int LayeredTree::update_upstream_evaluations(TreeKey node)
{
    /* overload allowing function to be called with a single node only */

    std::vector<TreeKey> node_in_vec(1, node);
    return update_upstream_evaluations(node_in_vec);
}

int LayeredTree::update_upstream_evaluations(std::vector<TreeKey> id_list)
{
    /* traverse the tree from a selection of keys. Returns the best evaluation
    at the root node */

    // std::vector<TreeKey> id_list = evaluated_ids_; //added_ids_; //old_ids_;
    std::vector<TreeKey> next_id_list;

    if (id_list.size() == 0) {
        std::cout << "update_upstream_evaluations() warning: id_list.size() == 0\n";
        // get the best evaluation at the root anyway
        std::vector<MoveEntry> best_moves = get_best_moves();
        if (best_moves.size() == 0) {
            throw std::runtime_error("update_upstream_evaluations() error: id_list.size() == 0 and root has no best moves");
        }
        int best_eval = best_moves[0].new_eval;
        return best_eval;
    }

    int num_layers = id_list[0].layer;
    std::vector<int> evals_layerwise(num_layers);

    if (num_layers > layer_pointers_.size() - 1) {
        std::cout << "node.layer = " << num_layers << ", layer_pointers_.size() - 1 = "
            << layer_pointers_.size() - 1 << "\n";
        throw std::runtime_error("LayeredTree::update_single_upstream_evaluations() error: node.layer exceeds number of layers");
    }

    std::shared_ptr<TreeLayer> this_layer_p;
    std::shared_ptr<TreeLayer> next_layer_p;
    int best_root_evaluation = 123456789;

    // we cascade through all layers except the very bottom, which is unfinished
    // int depth = layer_pointers_.size() - 2;
    int depth = num_layers;
    int depth_evaluated_to = num_layers + 1; // we update from parents of leaf nodes

    // testing: active move is deepest depth
    active_move_ = depth_evaluated_to;

    bool print_cascade = false; // for testing
    if (print_cascade) {
        std::cout << "Cascade depth is " << depth << ", depth evaluated to is "
            << depth_evaluated_to
            << ", initial length of id_list = " << id_list.size()
            << ". First element is:\n";
        for (int i = 0; i < 1; i++) {
            id_list[i].print();
        }
        std::cout << std::flush;
    }

    // loop backwards through the layers
    for (int k = 0; k <= depth; k++) {

        if (id_list.size() == 0) {
            throw std::runtime_error("update_upstream_evaluations() error: id_list.size() == 0");
        }

        int layer = depth - k + current_move_;
        this_layer_p = get_layer_pointer(layer);

        int sign = 2 * this_layer_p->white_to_play - 1;

        int offset = depth - k;              // distance from top of tree

        if (offset != 0) {
            // get the pointer to the next/previous layer
            next_layer_p = get_layer_pointer(layer - 1);
        }

        // are the below two needed?? Check this
        // add checkmate and drawn game nodes to the list
        this_layer_p->add_finished_games(id_list);
        // remove duplicates from the list
        id_list = this_layer_p->remove_duplicates(id_list);

        // for testing only
        if (print_cascade) {
            std::cout << "Update evaluations, length of id_list = " << id_list.size()
                << ", k = " << k << "; " << std::flush;
        }
        int test_ind = 0; // also for above testing

        // loop through the id items
        for (TreeKey& id_item : id_list) {

            if (print_cascade) {
                std::cout << test_ind << " " << std::flush;
                test_ind += 1;
            }

            // // check that all ids have the same layer
            // if (k == 0) {
            //     if (id_item.layer != num_layers) {
            //         std::cout << "id_item.layer = " << id_item.layer 
            //             << ", num layers expected = " << num_layers << "\n";
            //         throw std::runtime_error("id_list does not have all the same layers");
            //     }
            // }

            // if this id is in a downstream layer, skip it until we reach that layer
            if (id_item.layer != offset) {
                std::cout << "Ignorning!! id_item in a higher layer. This offset = " << offset
                    << ", this id_item = ";
                id_item.print();

                // // TESTING IGNORE HIGHER LAYERS SINCE THEIR EVALUATIONS DON'T REACH THE REQUIRED DEPTH
                // next_id_list.push_back(id_item);

                continue;
            }

            // set this node as active
            this_layer_p->board_list[id_item.entry].active_move = depth_evaluated_to;

            // find the best evaluation at this node
            NodeEval node = this_layer_p->find_max_eval(id_item.entry, sign, 
                depth_evaluated_to, offset);

            if (not node.active) {
                id_item.print();
                std::cout << "depth = " << depth << "\n";
                throw std::runtime_error("update_upstream_evaluations() error: node not active, but I have just set it active");
                continue;
            }

            // now we have the best evaluation at this node

            // overwrite the evaluation of this node
            this_layer_p->board_list[id_item.entry].eval = node.max_eval;

            // end here if on the final loop
            if (offset == 0) {
                if (id_list.size() != 1) {
                    throw std::runtime_error("id_list not == 1 on final cascade");
                }
                best_root_evaluation = node.max_eval;
                break;
            }

            // update this evaluation and activity in each parent
            for (const TreeKey& parent : this_layer_p->
                    board_list[id_item.entry].parent_keys) {

                next_layer_p->board_list[parent.entry].move_list[parent.move_index]
                    .new_eval = node.max_eval;

                // update parent activity in move list
                // next_layer_p->board_list[parent.entry].move_list[parent.move_index]
                //     .active_move += 1; // old
                next_layer_p->board_list[parent.entry].move_list[parent.move_index]
                    .active_move = depth_evaluated_to;

                // add the parent to the next id list
                next_id_list.push_back(parent);
            }      
        }

        // now we have finished with this id list
        id_list = next_id_list;
        //next_id_list.clear();
        std::vector<TreeKey>().swap(next_id_list); // swap with empty vector

        // loop to next layer up
    }


    // wipe the list of ids which we have now finished updating upstream of
    std::vector<TreeKey>().swap(added_ids_); // these are not used in this function
    // std::vector<TreeKey>().swap(evaluated_ids_); // comment this when using vector function input instead

    return best_root_evaluation;
}

TreeKey LayeredTree::find_hash(std::size_t hash, int layer)
{
    /* find a hash from a specific layer */
    TreeKey node_id;
    node_id.layer = -1;
    node_id.entry = -1;
    node_id.move_index = -1;
    node_id.evaluation = -1;

    std::shared_ptr<TreeLayer> layer_p = get_layer_pointer(layer);

    int key = layer_p->find_hash(hash);

    // if the hash exists in the layer
    if (key != -1) {
        node_id.evaluation = layer_p->board_list[key].eval;
    }

    node_id.layer = layer;
    node_id.entry = key;

    return node_id;
}

TreeKey LayeredTree::search_hash(std::size_t hash)
{
    /* find a hash from within the entire tree (layer unknown) */

    // search in every layer
    for (int i = 0; i < layer_pointers_.size(); i++) {
        TreeKey node_id = find_hash(hash, i);

        // if we have a match, return the node id
        if (node_id.entry != -1) {
            return node_id;
        }
    }

    // else there was no match, the hash was not found
    TreeKey node_not_found;
    node_not_found.layer = -1;
    node_not_found.entry = -1;
    node_not_found.move_index = -1;

    return node_not_found;
}

Engine::Engine()
{
    /* constructor for chess engine */

    // initialise with default settings
    settings.width = 10;
    settings.depth = 6;
    settings.first_layer_width_multiplier = 2;
    settings.prune_target = 2000;
    settings.num_cpus = 4;
    
    // new
    settings.eval_slack = 300; // this is 0.3 pawns
    settings.allowable_time_ms = 5000; // 5 seconds, default

    details.ms_per_board = -1; // to indicate not set
    
    analysis.allowable_time_exceeded = false;
    analysis.current_depth = 0;

    print_level = 2; // default
}

void Engine::set_width(int width)
{
    settings.width = width;
}

void Engine::set_depth(int depth)
{
    settings.depth = depth;
}

void Engine::set_prune_target(int prune_target)
{
    settings.prune_target = prune_target;
}

void Engine::set_first_layer_width_multiplier(int flwm)
{
    settings.first_layer_width_multiplier = flwm;
}

void Engine::print_startup(std::unique_ptr<LayeredTree>& tree_ptr)
{
    /* print a roundup of the engine's work. Print levels:
        0 - print nothing at all
        1 - print the timings and only the best move
        2 - print the timings and all of the best moves
        3 - print timings, best moves, and layer by layer updates
        4 - print timings, best moves, layer by layer, and best reponses
    */

    if (print_level > 0) {
        print_str("Engine generating moves to depth " + std::to_string(settings.depth));
    }

    if (print_level > 1) {
        print_vector(settings.depth_vector, "\tThe depth vector");
        print_vector(settings.width_vector, "\tThe width vector");
        print_vector(settings.prune_vector, "\tThe prune vector");
        std::cout << "\tuse_nn_eval = " << use_nn_eval << "\n";
    }

    if (print_level > 2) {
        if (settings.target_time < 0) {
            print_str("No target time or target boards");
        }
        else {
            std::string to_print = "Target time: " + std::to_string(settings.target_time) 
                + ", target boards: "
                + std::to_string(settings.target_boards) + " (board ms: "
                + std::to_string(details.ms_per_board) + " / per second: "
                + std::to_string(1000.0 / details.ms_per_board) + ")";
            print_str(to_print);
        }
    }
}

void Engine::print_layer(std::unique_ptr<LayeredTree>& tree_ptr, int layer)
{
    /* print a roundup of the engine's work. Print levels:
        0 - print nothing at all
        1 - print the timings and only the best move
        2 - print the timings and all of the best moves
        3 - print timings, best moves, and layer by layer updates
        4 - print timings, best moves, layer by layer, and best reponses
    */

    if (print_level == 1 or print_level == 2) {
        if (layer == 0) {
            print_str("Now searching at layer: 1 ", false); // no newline
        }
        else {
            print_str(", " + std::to_string(layer + 1), false);
        }

        // if we are on the last layer
        if (layer == settings.depth - 1) {
            print_str(" (final layer)");
        }
    }

    if (print_level > 2) {
        // unless at the start, print the best moves from the prev. layer
        if (layer > 0) {
            bool dead_nodes = true;
            tree_ptr->print_best_moves(dead_nodes);
        }
        // print an update that we are on a new layer
        print_str("Layer " + std::to_string(layer + 1));
    }
}

void Engine::print_roundup(std::unique_ptr<LayeredTree>& tree_ptr)
{
    /* print a roundup of the engine's work. Print levels:
        0 - print nothing at all
        1 - print the timings and only the best move
        2 - print the timings and all of the best moves
        3 - print timings, best moves, and layer by layer updates
        4 - print timings, best moves, layer by layer, and best reponses
    */

    if (print_level > 0) {
        print_str("\nThe engine has finished searching with the following details:");
        print_str("\tBest evaluation: " + std::to_string(analysis.best_eval * 1e-3));
        print_str("\tDepth checked to: " + std::to_string(analysis.depth_evaluated));
        print_str("\tBoards checked: " + std::to_string(details.boards_checked));
        print_str("\tTotal time (s): " + std::to_string(std::round(double(details.total_ms)) * 1e-3));
        print_str("\tTime per board (ms): " + std::to_string(std::round(details.ms_per_board * 1e3) * 1e-3));
        print_str("\tBoards per second: " + std::to_string(std::round(1000.0 / details.ms_per_board)));
        print_str("\tEngine ran out of time: " + std::to_string(analysis.allowable_time_exceeded));
        print_str("\tTarget depth: " + std::to_string(settings.depth));
        print_str("\tTarget width: " + std::to_string(settings.width));
    }

    if (print_level > 0) {
        tree_ptr->print_best_move();
    }

    if (print_level > 1) {
        bool print_dead_nodes = true;
        tree_ptr->print_best_moves(print_dead_nodes);
    }
    
    if (print_level > 3) {
        print_responses(tree_ptr);
    }

    std::cout << "Finished the print_roundup() function\n" << std::flush;
}

void Engine::print_responses(std::unique_ptr<LayeredTree>& tree_ptr)
{
    /* print what the engine considers the best response to its move */

    // get the move the engine is recommending
    std::vector<MoveEntry> best_moves = tree_ptr->get_best_moves();

    // get the key of the best move
    std::size_t hash = best_moves[0].new_hash;
    TreeKey response_key = tree_ptr->find_hash(hash, 1);

    if (response_key.entry == -1) {
        throw std::runtime_error("hash not found");
    }

    // now print the best response to this
    print_str("The engine thinks the best responses to its move are:");
    tree_ptr->print_best_moves(response_key, true);
}

void Engine::calculate_settings(double target_time)
{
    /* Decides how much searching the engine will do */

    constexpr int min_prune = 50;
    constexpr int min_width = 5;
    constexpr double mult_width[4] = { 2, 2, 1, 1 };
    constexpr double width_power = 2;
    constexpr double width_scale = 10;
    constexpr double wd_ratio = 1.0;
    constexpr double max_error = 0.1;
    constexpr double min_error = -0.1;

    /* to calculate the depth we use the formula:
        total_boards = depth * width_scale * (width) ^ width_power 
       
       We have two unknowns, depth and width. We relate them approximately:
        wd_ratio ~= width / depth

       We iterate to get a solution that fulfills both equations close enough
    */

    settings.width_vector.clear();
    settings.prune_vector.clear();
    settings.target_time = target_time;
    settings.allowable_time_ms = int(target_time * 1e3);
    std::cout << "The target time in seconds is " << settings.target_time << "\n";

    // if we do not have a set target time (negative value)
    if (target_time < 0 or details.ms_per_board < 0) {
        for (int i = 0; i < settings.depth; i++) {
            settings.width_vector.push_back(settings.width);
            settings.prune_vector.push_back(settings.prune_target);
        }
        settings.target_boards = -1;
        return;
    }

    // for testing
    return;

    // how many boards do we have time to search
    std::cout << "target time is: " << target_time <<
        ", ms_per_board is: " << details.ms_per_board << '\n';
    int total_boards = (1000 * target_time) / details.ms_per_board;

    // run a default search that is as small as reasonably possible
    if (total_boards < min_prune) {
        settings.depth = 2;
        settings.width_vector = std::vector<int>{ 7, 7 };
        settings.prune_vector = std::vector<int>{ 0, 0 };
    }

    double error = 100; // initialise to large value

    int depth = 5;
    int width = depth * wd_ratio;
    int sum_changes = 0;
    int old_sum_changes = -1;
    int old_old_sum_changes = -2;
    bool looping = false;
    double boards;
    double old_error;

    // first, coarse adjustment of the width and depth
    for (int i = 0; i < 10; i++) {

        // calculate the total boards with these settings
        boards = 0;
        for (int d = 0; d < depth; d++) {
            boards += min_prune + d * width_scale * std::pow(width, width_power);
        }
        
        // determine the error fraction
        error = (total_boards - boards) / total_boards;

        std::cout << "Total boards is: " << total_boards
            << ", depth is: " << depth << ", width is: " << width
            << ", expected boards is: " << boards
            << ", error is: " << error << '\n';

        // break upon acceptable error
        if (error < max_error and error > min_error) {
            break;
        }

        // check if we are in a loop
        if (old_old_sum_changes == old_sum_changes) {
            looping = true;
            std::cout << "looping is true\n";
        }

        // if we are looping and +ve error, break
        if (looping and abs(old_error) > abs(error)) {
            break;
        }

        // adjust the depth and width in line with error
        if (error > 0) {
            depth += 1;
            sum_changes += 1;
        }
        else {
            depth -= 1;
            sum_changes -= 1;
        }
        width = int(depth * wd_ratio);

        // save, to check we aren't looping
        old_old_sum_changes = old_sum_changes;
        old_sum_changes = sum_changes;
        old_error = error;
    }

    // now scale up the pruning amount to meet our target if needed
    int board_error = total_boards - boards;
    int target_error = 100;
    double tune = 1.0;
    double sum_t1 = 0, sum_t2 = 0, sum_t3 = 0;
    
    for (int i = 0; i < 50; i++) {

        if (abs(board_error) < target_error) {
            break;
        }

        boards = 0;
        if (board_error > 0) tune += 0.05;
        else tune -= 0.05;

        for (int d = 0; d < depth; d++) {
            boards += int(tune * 
                (min_prune + d * width_scale * std::pow(width, width_power)));
        }

        board_error = total_boards - boards;

        std::cout << "Total boards is: " << total_boards
            << ", depth is: " << depth << ", width is: " << width
            << ", expected boards is: " << boards
            << ", tune factor is: " << tune
            << ", error is: " << board_error << '\n';
    }
    

    // now we create our settings
    settings.width_vector.clear();
    settings.prune_vector.clear();

    for (int d = 0; d < depth; d++) {

        // set the width
        if (d < 4) {
            settings.width_vector.push_back(int(width * mult_width[d]));
        }
        else {
            settings.width_vector.push_back(width);
        }

        // set the prune target for that depth
        settings.prune_vector.push_back(int(tune * (min_prune 
            + d * width_scale * std::pow(width, width_power))));
    }

    // helpful to total up the target number of boards
    settings.target_boards = 0;
    for (int p : settings.prune_vector) { settings.target_boards += p; }

    settings.width = width;
    settings.depth = depth;

    std::cout << "The end of calculating:\n";
    print_vector(settings.width_vector, "width vector");
    print_vector(settings.prune_vector, "prune vector");
}

void Engine::calibrate(Board board, bool white_to_play = true)
{
    /* Estimate the ms per board */

    // create a copy of the current settings
    Settings copy = settings;

    // set some default settings
    settings.width = 10;
    settings.depth = 3;
    settings.prune_target = 0;

    // run generate, which calculates the ms_per_board at the end
    generate(board, white_to_play, -1);

    // restore the previous settings
    settings = copy;
}

void Engine::recursive_search(std::unique_ptr<LayeredTree>& tree_ptr,
    std::vector<TreeKey> best_replies)
{
    /* recursively called function which performs a depth first search of the
    move tree */

    if (best_replies.size() == 0) {
        std::cout << "recursive_search() found termination board state"
            << ", evaluation not available. I need to add a fix for this\n";
        throw std::runtime_error("best_replies.size() == 0 in recursive_search()");
    }

    // this is what we think is the 'best' so far on this line (ie the best reply)
    TreeKey node = best_replies[0];

    analysis.current_depth = best_replies[0].layer;

    std::cout << "Entered the recursive_search() function, layer pointer size = "
        << tree_ptr->layer_pointers_.size()
        << ", current layer (and current depth) = " << best_replies[0].layer << "\n";

    // check that we do not exceed the time limit
    if (analysis.allowable_time_exceeded) {
        return;
    }
    std::chrono::time_point<std::chrono::steady_clock> time_now = std::chrono::steady_clock::now();
    long current_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(time_now - start_).count();
    if (current_ms > settings.allowable_time_ms) {
        analysis.allowable_time_exceeded = true;
        // if we didn't get a single best evaluation, we must use this one
        if (analysis.best_move_sequence.size() == 0) {
            // first cascade the evaluations all the way up
            tree_ptr->update_upstream_evaluations();

            // now get the best evaluation at the root
            std::vector<MoveEntry> best_moves = tree_ptr->get_best_moves();
            int best_eval = best_moves[0].new_eval;

            // this evaluation is our best estimate
            analysis.best_eval = node.evaluation;
            analysis.best_move_sequence = tree_ptr->layer_pointers_[node.layer]
                                            ->board_list[node.entry].parent_moves;
            analysis.depth_evaluated = analysis.current_depth;
        }

        std::cout << "Hit the time target at depth " << analysis.current_depth
            << ", this evaluation = " << node.evaluation
            << ", best evaluation = " << analysis.best_eval << "\n";

        return;
    }

    // check we do not exceed the depth limit
    if (analysis.current_depth > settings.depth) {
        throw std::runtime_error("current depth deeper than settings.depth allows");
    }
    else if (analysis.current_depth == settings.depth) {
        // we have hit our depth limit, check if the current evaluation is better

        // first cascade the evaluations all the way up
        if (tree_ptr->evaluated_ids_.size() == 0) {
            tree_ptr->evaluated_ids_.push_back(node);
        }
        tree_ptr->update_upstream_evaluations();

        // now get the best evaluation at the root
        std::vector<MoveEntry> best_moves = tree_ptr->get_best_moves();
        int best_eval = best_moves[0].new_eval;

        if (best_eval * analysis.sign > analysis.best_eval * analysis.sign) {
            // the new evaluation is better
            analysis.best_eval = best_eval;
            analysis.best_move_sequence = tree_ptr->layer_pointers_[node.layer]
                                            ->board_list[node.entry].parent_moves;
            analysis.depth_evaluated = analysis.current_depth;
        }

        std::cout << "Hit target depth at " << analysis.current_depth
            << ", this evaluation = " << node.evaluation
            << ", best evaluation = " << analysis.best_eval << "\n";

        return;
    }

    /* ----- pruning ----- */

    // // if it is our move 
    // if (analysis.current_depth % 2 == 0) {
    //     if (node.evaluation * analysis.sign < 
    //         analysis.best_eval * analysis.sign - settings.eval_slack) {
    //         std::cout << "Prune our move at depth " << analysis.current_depth
    //         << ", this evaluation = " << node.evaluation
    //         << ", best evaluation = " << analysis.best_eval << "\n";     
    //         return;
    //     }
    // }
    // // it is their move
    // else {
    //     if (node.evaluation * analysis.sign * -1 < 
    //         analysis.best_eval * analysis.sign * -1 - settings.eval_slack) {
    //         std::cout << "Prune their move at depth " << analysis.current_depth
    //         << ", this evaluation = " << node.evaluation
    //         << ", their best evaluation = " << analysis.their_best_eval << "\n";     
    //         return;
    //     }
    // }

    // check if the evaluation of this line has dropped, and must be pruned
    if (node.evaluation * analysis.sign < 
            analysis.best_eval * analysis.sign - settings.eval_slack) {

        std::cout << "Prune at depth " << analysis.current_depth
            << ", this evaluation = " << node.evaluation
            << ", best evaluation = " << analysis.best_eval << "\n";     
        return;
    }

    // if here, we will now go deeper on this node

    // // get the best replies to the given node position
    // std::vector<MoveEntry> best_replies = tree_ptr->get_best_moves(node);

    std::cout << "Going deeper, into depth " << analysis.current_depth + 1 << "\n";

    // for the set width, determine the subsequent best responses to these best replies
    for (int i = 0; i < settings.width; i++) {

        std::cout << "Layer to add depth to is " << best_replies[i].layer
            << ", recursive_search() i = " << i << "\n";
        for (int j = 0; j < best_replies.size(); j++) {
            best_replies[j].print();
        }

        std::vector<TreeKey> reply_keys = tree_ptr->add_depth_at_key(best_replies[i]);

        // now recursively call this function to progress each to the next depth
        recursive_search(tree_ptr, reply_keys);

        /* ----- updating best evaluations ----- */

        // if it is our move at this depth
        if (analysis.current_depth % 2 == 0) {

        }
        else { // their move
        
        }

        // check if we are out of time
        if (analysis.allowable_time_exceeded) {
            return;
        }
    }

    std::cout << "Retreating to previous depth of " << analysis.current_depth - 1 << "\n";

    return; // finished the whole width, recursion may continue
}

void Engine::depth_search(std::unique_ptr<LayeredTree>& tree_ptr)
{
    /* run the depth search with default depth and width */

    depth_search(tree_ptr, settings.depth, settings.width);
}

void Engine::depth_search(std::unique_ptr<LayeredTree>& tree_ptr, int depth, int width)
{
    /* do a depth first search of the board. Depth and width should be set */
    
    print_str("Entered the depth_search() function: depth = "
        + std::to_string(depth) + ", width = " + std::to_string(width));

    if (analysis.allowable_time_exceeded) {
        print_str("Allowable time exceeded, cancelling any further depth search");
        return;
    }

    // set the width in the tree
    tree_ptr->width_ = width;
    settings.width = width;
    settings.depth = depth;

    // how to handle printing during search?
    // print_layer(tree_ptr, 0);

    // determine if the game continues
    bool game_continues = tree_ptr->check_game_continues();

    if (not game_continues) {
        print_str("Engine::depth_search() found game_continues=false in initial board");
        return;
    }

    // this function uses a stack to manage the tree search
    std::stack<TreeKey> key_stack;

    // what move are we starting on
    int first_layer_move = 1; // first move

    // add the root position as the first element in the stack
    key_stack.push(tree_ptr->root_);

    bool print_debug = false;

    // continue to work the stack
    while (not key_stack.empty()) {

        if (print_debug) std::cout << "Stack size = " << key_stack.size() << "\n";

        // get the next destination node
        TreeKey node = key_stack.top();
        key_stack.pop(); // delete this entry after getting its data

        // determine the current depth we are at (from this node)
        analysis.current_depth = node.layer;

        /* ----- time limits and safety checks ----- */

        std::chrono::time_point<std::chrono::steady_clock> time_now = std::chrono::steady_clock::now();
        long current_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(time_now - start_).count();
        if (current_ms > settings.allowable_time_ms) {
            analysis.allowable_time_exceeded = true;

            // update the evaluations from the parent (from this node causes an error)
            std::vector<TreeKey> node_parents = tree_ptr->layer_pointers_[node.layer]
                                                    ->board_list[node.entry].parent_keys;
            int best_eval = tree_ptr->update_upstream_evaluations(node_parents);

            // check if this line is our best evaluation
            if (best_eval * analysis.sign > analysis.best_eval * analysis.sign) {
                // the new evaluation is better
                analysis.best_eval = best_eval;
                analysis.best_move_sequence = tree_ptr->layer_pointers_[node.layer]
                                                ->board_list[node.entry].parent_moves;
                analysis.depth_evaluated = analysis.current_depth;
            }

            if (print_debug) {
                std::cout << "Hit the time target at depth " << analysis.current_depth
                    << ", this evaluation = " << node.evaluation
                    << ", best evaluation = " << analysis.best_eval << "\n";
            }

            break; // exit the depth search loop
        }
        else {
            analysis.allowable_time_exceeded = false;
        }

        if (analysis.current_depth >= settings.depth) {
            throw std::runtime_error("current depth deeper than settings.depth allows");
        }

        /* ----- branch pruning ----- */ 

        // can I use 'best_moves' at the current node? with 'active move' set to current depth?
        if (false) { // new idea, not tested or finished yet
            if (analysis.current_depth % 2 == 0) { // if it is our move
                if (node.evaluation * analysis.sign < 
                    analysis.best_eval * analysis.sign - settings.eval_slack) {
                    std::cout << "Prune our move at depth " << analysis.current_depth
                    << ", this evaluation = " << node.evaluation
                    << ", best evaluation = " << analysis.best_eval << "\n";     
                    continue; // do not grow the tree from this node
                }
            }
            // it is their move
            else {
                if (node.evaluation * analysis.sign * -1 < 
                    analysis.best_eval * analysis.sign * -1 - settings.eval_slack) {
                    std::cout << "Prune their move at depth " << analysis.current_depth
                    << ", this evaluation = " << node.evaluation
                    << ", their best evaluation = " << analysis.their_best_eval << "\n";     
                    continue; // do not grow the tree from this node
                }
            }
        }
        else { // old method, reasonably reliable first go
            // check if the evaluation of this line has dropped, and must be pruned
            if (node.evaluation * analysis.sign < 
                    analysis.best_eval * analysis.sign - settings.eval_slack) {

                if (print_debug) {
                    std::cout << "Prune at depth " << analysis.current_depth
                        << ", this evaluation = " << node.evaluation
                        << ", best evaluation = " << analysis.best_eval 
                        << ", eval slack = " << settings.eval_slack
                        << ", node.eval*sign = " << node.evaluation * analysis.sign
                        << " < best_eval*sign - slack = "
                        << analysis.best_eval * analysis.sign - settings.eval_slack
                        << "\n";  
                }   
                continue; // do not grow the tree from this node
            }
        }

        /* ----- tree expansion ----- */

        // generate the best replies to the state at the current node
        if (print_debug) {
            std::cout << "Layer is " << node.layer << "\n" << std::flush;
            node.print(); std::cout << std::flush;
        }
        std::vector<TreeKey> reply_keys = tree_ptr->add_depth_at_key(node);

        // check if this is a termination
        if (reply_keys.size() == 0) {
            //...
            // throw std::runtime_error("reply_keys.size() == 0 in depth_search");
            continue;
        }

        // check if these new entries have reached the maximum depth
        if (reply_keys[0].layer >= depth) {

            // update what is our new best evaluation
            analysis.best_eval =  tree_ptr->update_upstream_evaluations(node);

            // // update evaluations upstream of this leaf node
            // int best_eval =  tree_ptr->update_upstream_evaluations(node);

            // // determine if this line has improved the best evaluation
            // if (best_eval * analysis.sign > analysis.best_eval * analysis.sign) {

            //     // the new evaluation is better
            //     analysis.best_eval = best_eval;
            //     analysis.best_move_sequence = tree_ptr->layer_pointers_[node.layer]
            //                                     ->board_list[node.entry].parent_moves;
            //     analysis.depth_evaluated = reply_keys[0].layer;
            // }

            continue; // do not go any deeper from this node
        }

        // prepare to add these replies to the stack, up to the maximum width
        int num_keys_to_add = width;
        if (reply_keys.size() < num_keys_to_add) {
            num_keys_to_add = reply_keys.size();
        }

        // add the first 'width' entries, but add them in reverse order
        for (int i = 0; i < num_keys_to_add; i++) {
            int ind = num_keys_to_add - 1 - i;
            key_stack.push(reply_keys[ind]);
        }

        /* ----- end of loop ----- */
    }

    // // once everything is finished, update any remaining evalutions
    // tree_ptr->update_upstream_evaluations();
}

std::vector<MoveEntry> Engine::generate_engine_moves(Board board, bool white_to_play, double target_time)
{
    /* generate using a depth first search of the board */

    print_str("Entered the generate() function");

    // start the clock
    start_ = std::chrono::steady_clock::now();

    // initialise settings
    analysis.allowable_time_exceeded = false;
    analysis.current_depth = 0;

    // testing: calculate parameters
    calculate_settings(target_time);

    // prepare analysis
    analysis.depth_evaluated = 0;
    analysis.this_eval_depth_evaluated = 0;
    analysis.best_eval_layerwise = std::vector<int>(settings.depth);
    analysis.this_eval_layerwise = std::vector<int>(settings.depth);

    if (white_to_play) {
        // we are white
        analysis.white = true;
        analysis.sign = 1;
        analysis.best_eval = WHITE_MATED; // our worst possible evaluation
        analysis.their_best_eval = BLACK_MATED; // their worst possible evaluation
    }
    else {
        // we are black
        analysis.white = false;
        analysis.sign = -1;
        analysis.best_eval = BLACK_MATED; // our worst possible evaluation
        analysis.their_best_eval = WHITE_MATED; // their worst possible evaluation
    }

    // populate the 'best evals' with the worst possible case at each layer
    for (int i = 0; i < settings.depth; i++) {
        if (i % 2 == 0) { // our move
            analysis.best_eval_layerwise[i] = WHITE_MATED * analysis.sign;
            analysis.this_eval_layerwise[i] = WHITE_MATED * analysis.sign;
        }
        else { // their move
            analysis.best_eval_layerwise[i] = BLACK_MATED * analysis.sign;
            analysis.this_eval_layerwise[i] = BLACK_MATED * analysis.sign;
        }
    }

    // make a new tree
    std::unique_ptr<LayeredTree> tree_ptr = std::make_unique<LayeredTree>
        (board, white_to_play, settings.width);
    tree_ptr->use_nn_eval = use_nn_eval;

    // temporary measure: hardcode the search parameters
    std::vector<int> depth_vector {2, 4, 6, 8};
    std::vector<int> width_vector {30, 15, 10, 5};
    std::vector<int> prune_slack_vector {10000, 1000, 500, 250};

    // std::vector<int> depth_vector {1};
    // std::vector<int> width_vector {40};

    // ensure the settings correctly record the search parameters
    settings.depth = depth_vector[depth_vector.size() - 1];
    settings.depth_vector = depth_vector;
    settings.width_vector = width_vector;
    settings.prune_vector = prune_slack_vector;

    // print initial information
    print_startup(tree_ptr);

    // depth search with iterative deepening
    for (int i = 0; i < depth_vector.size(); i++) {
        settings.eval_slack = prune_slack_vector[i];
        depth_search(tree_ptr, depth_vector[i], width_vector[i]);
    }

    print_str("Depth search has now finished. Now extracting the best moves.");

    // reduce the active move
    tree_ptr->active_move_ = depth_vector[0];
    if (tree_ptr->active_move_ > analysis.depth_evaluated) {
        std::cout << "Engine::generate() warning: active_move_ = " 
            << tree_ptr->active_move_ << ", analysis.depth_evaluated = "
            << analysis.depth_evaluated << ", so capping active_move_\n";
        tree_ptr->active_move_ = analysis.depth_evaluated;
    }
    std::cout << "Active move is " << tree_ptr->active_move_ << "\n";

    std::vector<MoveEntry> best_moves = tree_ptr->get_best_moves();

    // end the clock
    end_ = std::chrono::steady_clock::now();

    // determine the timings overall and per board
    details.total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
    details.boards_checked = tree_ptr->boards_checked_;
    details.ms_per_board = double(details.total_ms) / double(details.boards_checked);

    // print out a roundup of the engine generation
    print_roundup(tree_ptr);

    if (best_moves.size() == 0) {
        throw std::runtime_error("no best moves, all nodes must be dead");
    }

    return best_moves;
}

Move Engine::generate(Board board, bool white_to_play, double target_time)
{
    /* generate a single best move, and return this */

    std::vector<MoveEntry> best_moves = generate_engine_moves(board, white_to_play, target_time);

    return best_moves[0].move;
}

std::vector<MoveEntry> Engine::generate_engine_moves_FEN(std::string fen, double target_time)
{
    /* generate a list of candidate moves, and their evaluations, from an fen position */

    Board board = FEN_to_board(fen);

    // extract who plays next from the board
    bool white_to_play;
    if (does_white_play_next(board)) {
        white_to_play = true;
    }
    else if (does_black_play_next(board)) {
        white_to_play = false;
    }
    else {
        std::cout << "FEN string: " << fen << "\n";
        throw std::runtime_error("Engine::generate_moves() error: who plays next was NOT extracted from fen string");
    }

    return generate_engine_moves(board, white_to_play, target_time);
}

void Engine::enable_nn_evaluator()
{
    /* turn on using a neural network evaluator, using the default path */

    enable_nn_evaluator(nn_eval_path);
}

void Engine::enable_nn_evaluator(std::string path)
{
    /* turn on using a neural network evaluator, and give the path from where it should be loaded */

    #if defined(LUKE_PYTORCH)

        // check if we have already loaded a neural network
        bool already_loaded = is_nn_loaded();
        if (already_loaded) {
            // check if the new given path is identical to our loaded path
            if (path == nn_eval_path) {
                std::cout << "Engine::enable_nn_evaluator(): nn_eval already loaded, not reloading, path is: " << path << "\n";
                std::cout << "Engine::enable_nn_evaluator() has set use_nn_eval = " << use_nn_eval << "\n";
                use_nn_eval = true;
                return;
            }
        }

        try {
            init_nn(path);
        }
        catch (const c10::Error& e) {
            std::cout << "Engine::enable_nn_evaluator() error: network could not be loaded, aborting\n";
        }

        use_nn_eval = is_nn_loaded();

        // if successful, update the loadpath
        if (use_nn_eval) {
            nn_eval_path = path;
        }

    #else

        std::cout << "Engine::enable_nn_evaluator() error: not compiled with pytorch, aborting, nn_eval NOT AVAILABLE\n";
        use_nn_eval = false;

    #endif

    std::cout << "Engine::enable_nn_evaluator() has set use_nn_eval = " << use_nn_eval << "\n";
}

void Engine::disable_nn_evaluator()
{
    /* disable use of the nn evaluator */

    use_nn_eval = false;
    std::cout << "Engine::disable_nn_evaluator() has set use_nn_eval = " << use_nn_eval << "\n";
}

GameBoard::GameBoard()
{
    /* constructor */

    init();
}

GameBoard::GameBoard(Board board, bool white_to_play)
{
    /* overload */

    init(board, white_to_play);
}

GameBoard::GameBoard(std::vector<std::string> mymove_list)
{
    init(mymove_list);
}

void GameBoard::init()
{
    /* initialise with default values - empty board */

    Board empty_board = create_board();
    bool white_first = true;
    move_list.clear();

    init(empty_board, white_first);
}

void GameBoard::init(std::vector<std::string> mymove_list)
{
    /* overload */

    // first, start with an empty board
    Board board = create_board();
    bool white_to_play = true;

    // now loop through the moves, and apply them to the board
    for (int i = 0; i < mymove_list.size(); i++) {

        // check the move is legal, then if so, make it on the board
        verified_move move_numbers = verify_move(board, white_to_play, mymove_list[i]);
        if (move_numbers.legal) {
            make_move(board, move_numbers.start_sq, move_numbers.dest_sq,
                move_numbers.move_mod);
        }
        else {
            std::string error_msg = "The move " + mymove_list[i] + " is not legal!";
            std::vector<std::string> mymove_list =
                std::vector<std::string>(mymove_list.begin(), mymove_list.begin() + i);
            print_vector(mymove_list, "The move_list used was");
            break;
        }

        white_to_play = not white_to_play;
    }

    move_list = mymove_list;

    init(board, white_to_play);
}

void GameBoard::init(Board myboard, bool mywhite_to_play)
{
    /* initialise the board - all init functions lead here */

    board = myboard;
    white_to_play = mywhite_to_play;

    // check the outcome of the board
    total_legal_moves_struct tlm = total_legal_moves(board, white_to_play);
    outcome = tlm.outcome;

    // create a chess engine object
    engine_pointer = std::make_unique<Engine>();

    // // calibrate the engine (but print nothing)
    // engine_pointer->print_level = 0;
    // engine_pointer->calibrate(board);

    // set the print level to 4 (options 0-4)
    engine_pointer->print_level = 2;
}

bool GameBoard::move(std::string move)
{
    /* make a move and return true if successful, false if not legal */

    // convert to number format and verify that the move is legal
    verified_move move_verify = verify_move(board, white_to_play, move);

    if (move_verify.legal) {
        // make the move on our internal board
        make_move(board, move_verify.start_sq, move_verify.dest_sq,
            move_verify.move_mod);

        // update our internal variables
        white_to_play = not white_to_play;
        move_list.push_back(move);

        // now update the outcome of the board after this move
        total_legal_moves_struct tlm = total_legal_moves(board, white_to_play);
        outcome = tlm.outcome;

        return true;
    }
    else {
        return false;
    }
}

void GameBoard::undo(int moves_undone = 1)
{
    /* undo a number of moves on the board */

    // adjust our internal varibales
    for (int i = 0; i < moves_undone; i++) {
        move_list.pop_back();
        white_to_play = not white_to_play;

        // check we haven't got back to a bare board
        if (move_list.size() == 0) {
            break;
        }
    }

    // create a new board using the move list
    board = create_board(move_list);

    // so safety, recheck the board outcome (it should always be = 0)
    total_legal_moves_struct tlm = total_legal_moves(board, white_to_play);
    outcome = tlm.outcome;

    if (outcome != 0) {
        throw std::runtime_error("board outcome not = 0 after undo!");
    }
}

void GameBoard::reset()
{
    /* reset the board */

    init();
}

void GameBoard::reset(std::vector<std::string> mymove_list)
{
    /* overload to reset to a specific board state */

    init(mymove_list);
}

void GameBoard::reset(Board board, bool white_to_play)
{
    /* overload to reset to a specific given board */

    move_list.clear();
    init(board, white_to_play);
}

bool GameBoard::check_promotion(std::string move)
{
    /* check if the given move is a promotion */

    return is_promotion(board, white_to_play, move);
}

int GameBoard::get_square_colour(std::string square)
{
    /* find out which colour player is on a given square */

    int square_number = square_letters_to_numbers(square);

    if (board.arr[square_number] <= -7 or
        board.arr[square_number] >= 7) {
        throw std::runtime_error("board error, square contains bad value");
    }
    
    if (board.arr[square_number] > 0) {
        return 1;
    }
    else if (board.arr[square_number] < 0) {
        return -1;
    }
    else {
        return 0;
    }
}

int GameBoard::get_square_raw_value(std::string square)
{
    int square_number = square_letters_to_numbers(square);
    return get_square_raw_value(square_number);
}

int GameBoard::get_square_raw_value(int square_num)
{
    if (square_num < 0 or square_num > 119) {
        throw std::runtime_error("square_num is out of bounds");
    }

    return board.arr[square_num];
}

std::string GameBoard::get_square_piece(std::string square)
{
    int square_number = square_letters_to_numbers(square);
    return get_square_piece(square_number);
}

std::string GameBoard::get_square_piece(int square_num)
{
    if (square_num < 0 or square_num > 119) {
        throw std::runtime_error("square_num is out of bounds");
    }

    int raw_value = board.arr[square_num];

    switch (raw_value) {
    case -7: case 7: return "off board";
    case -6: return "black king";
    case -5: return "black queen";
    case -4: return "black rook";
    case -3: return "black bishop";
    case -2: return "black knight";
    case -1: return "black pawn";
    case 0: return "empty";
    case 1: return "white pawn";
    case 2: return "white knight";
    case 3: return "white bishop";
    case 4: return "white rook";
    case 5: return "white queen";
    case 6: return "white king";
    default:
        throw std::runtime_error("board value corrupted, not between -7 and 7");
    }
}

void GameBoard::use_nn_evaluator()
{
    /* try to load a neural network evaluator in the engine using the default path */

    engine_pointer->enable_nn_evaluator();
}

void GameBoard::use_nn_evaluator(std::string path_to_load_model)
{
    /* try to load a neural network evaluator in the engine */

    engine_pointer->enable_nn_evaluator(path_to_load_model);
}

std::string GameBoard::get_engine_move()
{
    /* overload */

    // run the engine without a time limit
    return get_engine_move(-1);
}

std::string GameBoard::get_engine_move(int target_time)
{
    /* The engine generates the next move on the board, and plays it.
    This function releases the python global interpreter lock */

    Move engine_move = engine_pointer->generate(board, white_to_play, target_time);

    return engine_move.to_letters();
}

std::string GameBoard::get_engine_move_no_GIL(int target_time)
{
    /* The engine generates the next move on the board, and plays it.
    This function releases the python global interpreter lock */

    return get_engine_move(target_time);
}

Game::Game()
{
    /* constructor */

    // create the chess engine object
    engine_pointer = std::make_unique<Engine>();
}

void Game::play_terminal(bool human_first)
{
    /* play a game */

    Board board = create_board();

    // engine settings
    engine_pointer->set_depth(6);
    engine_pointer->set_width(5);
    double target_time = 5;

    bool engine_next = not human_first;
    bool engine_colour = not human_first;
    Move next_move;

    while (true) {

        std::cout << "The board state is:\n";
        print_board(board, true);

        if (engine_next) {
            next_move = engine_pointer->generate(board, engine_colour, target_time);
        }
        else {
            next_move = get_human_move(board, not engine_colour);
        }

        if (quit) {
            std::cout << "Quiting out of program\n";
            break;
        }

        std::cout << "The move to be played next is " << next_move.to_letters() << '\n';
        make_move(board, next_move.start_sq, next_move.dest_sq, next_move.move_mod);
        
        engine_next = not engine_next;
        std::cout << ((engine_next) ? "The engine goes next\n" : "Human to play\n");
    }
}

Move Game::get_human_move(Board board, bool white_to_play)
{
    /* get a move from the human */

    std::string move_string;

    while (true) {

        std::cout << "Choose a move to play: ";

        // get the move
        std::cin >> move_string;

        if (move_string == "quit" or
            move_string == "Quit" or
            move_string == "exit" or
            move_string == "Exit" or
            move_string == "cancel" or
            move_string == "Cancel") {
            quit = true;
            Move empty_move;
            return empty_move;
        }

        // convert to number format and verify that the move is legal
        verified_move move_verify = verify_move(board, white_to_play,
            move_string);

        Move move;
        move.start_sq = move_verify.start_sq;
        move.dest_sq = move_verify.dest_sq;
        move.move_mod = move_verify.move_mod;

        if (move_verify.legal) {
            std::cout << "Move legal\n";
            return move;
        }
        else {
            std::cout << "Move is not legal, please try again. Type 'exit' to cancel\n";
        }
    }
}

bool Game::is_move_legal(Board board, bool white_to_play, std::string move_string)
{
    /* verify that a move is legal*/

    // convert to number format and verify that the move is legal
    verified_move move_verify = verify_move(board, white_to_play, move_string);

    Move move;
    move.start_sq = move_verify.start_sq;
    move.dest_sq = move_verify.dest_sq;
    move.move_mod = move_verify.move_mod;

    if (move_verify.legal) {
        return true;
    }
    else {
        return false;
    }
}

void Game::use_nn_evaluator()
{
    /* try to load a neural network evaluator in the engine using the default path */

    engine_pointer->enable_nn_evaluator();
}

void Game::use_nn_evaluator(std::string path_to_load_model)
{
    /* try to load a neural network evaluator in the engine */

    engine_pointer->enable_nn_evaluator(path_to_load_model);
}

Board Game::human_move(Board board, bool white_to_play, std::string move_string)
{
    /* make a move on the board, return the new board */

    // convert to number format and verify that the move is legal
    verified_move move_verify = verify_move(board, white_to_play, move_string);

    if (move_verify.legal) {
        make_move(board, move_verify.start_sq, move_verify.dest_sq,
            move_verify.move_mod);
    }
    else {
        throw std::runtime_error("human move is not legal!");
    }

    return board;
}

Board Game::engine_move(Board board, bool white_to_play)
{
    /* gets the engine to find a move, then returns the board after that move */

    Move engine_move;

    engine_move = engine_pointer->generate(board, white_to_play);

    make_move(board, engine_move.start_sq, engine_move.dest_sq, 
        engine_move.move_mod);

    moves.push_back(engine_move);
    move_letters.push_back(engine_move.to_letters());

    return board;
}

std::string Game::get_last_move()
{
    return move_letters[move_letters.size() - 1];
}

/* ------------------------------------------------------------------------- */