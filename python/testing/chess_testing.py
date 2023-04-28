
# the aim of this file is to complete unit tests for the chess engine

# my idea is to test every cpp function against the python equivalent, and the input to the test battery will
# be the starting board state. Therefore, you can input a large number of different boards and check that all
# of the functions are working correctly

# should I make this as a class? I think so

import cpp_py_conversions_vs as cpc
import board_func as bf
import board_light as bl
import time
#import engine_class as ec

class TestChess:
    def __init__(self, quick=False):
        """Constructor"""

        t0 = time.process_time()

        self.moves = [[]]
        self.moves[0] = ['f2f3', 'e7e5', 'g2g4', 'd8h4'] # tricky checkmate, fixed

        if not quick:
            self.moves.append(['e2e4', 'c7c6', 'e4e5', 'e7e6', 'd1h5', 'd8a5', 'g2g4', 'f8e7', 'a2a3', 'e7h4', 'g4g5', 'b7b6', 
              'b1c3', 'a7a6', 'c3a2', 'a8a7', 'b2b3', 'a7a8', 'c1b2', 'a8a7', 'a1d1', 'a7a8', 'b2c3', 'a8a7', 
              'd2d3', 'a7a8', 'c3d2', 'a8a7', 'g1f3', 'a5c5', 'd2c1', 'a7a8', 'f3d2', 'a8a7', 'f1g2', 'a7a8', 
              'h1f1', 'a8a7', 'h5h7', 'c5e3']) # pinned pawn checkmate
            self.moves.append(['e2e4', 'e7e5', 'f2f4', 'd8h4', 'e1e2', 'g8f6', 'g1f3', 'h4f4', 'b1c3', 'f8c5', 'g2g3', 'f4g4', 
              'd2d3', 'd7d5', 'c1d2', 'd5e4', 'd3e4', 'b8c6', 'c3d5', 'c8e6', 'c2c3', 'h7h6', 'd1a4', 'a8d8', 
              'a1d1', 'f6e4', 'd5c7', 'e8e7', 'e2e1', 'e4d2', 'a4g4', 'e6g4', 'f3d2', 'g4d1', 'e1d1', 'd8d7', 
              'f1b5', 'h8d8', 'c3c4', 'd7d2', 'd1c1', 'c5e3', 'b5c6', 'd2h2', 'c1b1', 'h2h1', 'b1c2', 'd8d2', 
              'c2b3', 'h1b1', 'c7d5', 'e7e6', 'd5e3', 'b7c6', 'e3g4', 'c6c5', 'b3a4', 'd2b2', 'a4a5', 'b2a2'])
                # entire game with lots of pins and dangerous king positions
            self.moves.append(['e2e4', 'd7d5', 'e4d5', 'c7c6', 'd5c6', 'g8f6', 'c6b7', 'e7e5', 'f1b5', 'c8d7', 'b7a8', 'd8a5', 
              'c2c4', 'f8b4', 'c4c5', 'f6e4', 'c5c6', 'a5a4', 'c6c7', 'b4e7', 'c7b8', 'e7d8', 'b8e5', 'e8f8', 
              'a2a3', 'f7f5', 'b2b3', 'f5f4', 'b3b4', 'f4f3', 'g1h3', 'f3g2', 'a8b8', 'e4g3', 'b8d6', 'f8g8', 
              'h3g5', 'g2h1']) # entire game with lots of pawn promotions and checks
            self.moves.append(['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5','c2c3', 
              'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4',
              'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4',
              'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5',
              'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2',
              'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6',
              'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6', 'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3', 'h4h5',
              'e7d7', 'c7c8', 'd7c8', 'a3a4', 'c8c7', 'b4b5', 'a6b5', 'a4b5', 'd6d5', 'b5b6', 'c7b6', 'g2f1',
              'd5d4', 'f1g2', 'b6c5', 'g2f1', 'c5c4', 'f1g2', 'd4d3', 'g2f1', 'e5e4', 'f1g2', 'd3d2', 'g2f1',
              'd2d1', 'f1g2', 'd1f3', 'g2h3', 'e4e3', 'h3h4']) # stockfish compare game

        self.py_boards = []
        self.cpp_boards = []
        self.to_play = []

        for move_set in self.moves:
            for i in range(len(move_set)):
                mini_set = move_set[:i+1]
                new_py, new_cpp = self.create_boards(mini_set)
                self.py_boards.append(new_py)
                self.cpp_boards.append(new_cpp)
                # whos move is it
                if i % 2 == 0:
                    self.to_play.append(False)
                else:
                    self.to_play.append(True)

        t1 = time.process_time()
        print("Time taken to create test object boards:", t1-t0, "seconds")
        print("Number of boards created", len(self.py_boards))

        return

    def print_boards(self, py_board, cpp_board):
        """This function prints the two boards"""

        print("The python board is")
        bl.print_board(py_board, True)
        print("The cpp board is")
        bf.print_board(cpp_board, False)

        return

    def compare_board(self, py_board, cpp_board):
        """This function takes two boards and verifies that they are exactly the same"""

        # first convert the python board to cpp
        py_as_cpp = cpc.board_to_cpp(py_board)

        # now compare the two boards
        is_identical = bf.are_boards_identical(cpp_board, py_as_cpp)

        return is_identical

    def create_boards(self, move_set):
        """This function creates a python and cpp board from a move set"""

        py_board = bl.create_board(move_set, True)
        cpp_board = cpc.board_to_cpp(py_board)

        return py_board, cpp_board

    def check_checkmate(self, py_board, cpp_board, white_to_play):
        """This function confirms that the cpp checkmate function gives the
        correct result"""

        cpp_mate = bf.test_checkmate(cpp_board, 2 * white_to_play - 1)
        py_mate = bl.checkmate(py_board, white_to_play)

        if cpp_mate != py_mate:
            print("Checkmate disagreement! Python ground truth says", py_mate, ", cpp says", cpp_mate)
            print("The board is")
            bf.print_board(cpp_board, False)
            return False

        # FOR TESTING
        if py_mate:
            print("checkmate agreement")
        #else:
        #    print("no checkmate agreement")

        return True

    def check_move(self, py_board, cpp_board, move):
        """This function checks that a move being made results in a cpp and py
        board which are both the same"""

        # copy the base board states
        py_copy = [x[:] for x in py_board]
        cpp_copy = bf.copy_board(cpp_board)

        start_sq = move[0]
        dest_sq = move[1]
        move_mod = move[2]

        bl.make_move(py_copy, [start_sq, dest_sq], move_mod)
        bf.make_move(cpp_copy, start_sq, dest_sq, move_mod)

        # confirm that these two new boards are the same
        is_same = self.compare_board(py_copy, cpp_copy)

        if not is_same:
            print("The move did not result in the same board for both python and cpp!")
            print("The move was", [start_sq, dest_sq, move_mod])
            self.print_boards(py_copy, cpp_copy)
            return False, None, None

        return True, py_copy, cpp_copy

    def check_evals(self, py_board, cpp_board, white_to_play):
        """This function compares evaluations using python and cpp"""

        cpc.eval_piece_roundup(py_board, cpp_board, white_to_play)

        return

    def compare_gen_moves_eval(self, base_board, move, white_to_play):
        """This function digs in to see why and the evaluation from gen_moves
        is different to that from eval_board"""

        move_board = move.get_board()
        start_sq = move.get_start_sq()
        dest_sq = move.get_dest_sq()
        move_mod = move.get_move_mod()

        eb_eval = bf.eval_board(move_board, not white_to_play)
        gm_eval = move.get_evaluation()

        base_tlm = bf.total_legal_moves(base_board, white_to_play)
        
        start_ind = (start_sq - 21) - 2 * ((start_sq // 10) - 2)
        dest_ind = (dest_sq - 21) - 2 * ((dest_sq // 10) - 2)

        start_view = base_tlm.get_piece_view(start_ind)
        dest_view = base_tlm.get_piece_view(dest_ind)

        if len(dest_view) == 0:
            temp_pad_struct = bf.piece_attack_defend(base_board, dest_sq, 6, 1)
            dest_view = temp_pad_struct.get_piece_view()

        # get the piece view sorted
        fv_start_view = bf.find_view(base_board, start_sq)
        fv_dest_view = bf.find_view(base_board, dest_sq)

        piece_view = dest_view[:]
        piece_view.append(dest_sq)

        for sq in start_view:
            if sq not in piece_view:
                piece_view.append(sq)
        if move_mod == 3:
            if white_to_play:
                if dest_sq - 10 not in piece_view:
                    piece_view.append(dest_sq - 10)
            else:
                if dest_sq + 10 not in piece_view:
                    piece_view.append(dest_sq + 10)
        if move_mod == 5:
            if dest_sq > start_sq:
                rook_sq = dest_sq - 1
            else:
                rook_sq = dest_sq + 1
            if rook_sq not in piece_view:
                piece_view.append(rook_sq)

        phase = bf.determine_phase(move_board, not white_to_play)
        old_phase = base_tlm.get_phase();
        new_phase = phase.get_phase();

        # begin
        new_eval = base_tlm.get_evaluation();
        new_eval += phase.get_eval_adjust() - base_tlm.get_phase_adjust()

        entire_board = []
        for i in range(8):
            for j in range(8):
                ij_index = (2 + i) * 10 + (j + 1);
                entire_board.append(ij_index)

        old_values = []
        new_values = []
        running_eval = phase.get_eval_adjust()

        #for view in piece_view:
        for view in entire_board:

            view_ind = (view - 21) - 2 * ((view // 10) - 2)
            old_value = base_tlm.get_old_value(view_ind)
            mate = False

            piece_type = move_board.look(view)
            if piece_type < 0:
                piece_colour = -1
                piece_type *= -1
            elif piece_type > 0: piece_colour = 1
            
            if piece_type == 0:
                new_value = 0
            else:
                # investigate the piece
                temp_pad = bf.piece_attack_defend(move_board, view,
                                                  piece_type, piece_colour)
                new_value = bf.eval_piece(move_board, not white_to_play, view,
                                         phase, piece_type, piece_colour, mate,
                                         temp_pad)
                # IGNORE CHECKMATE

            if view in piece_view:
                new_eval += new_value - old_value;

            running_eval += new_value

            old_values.append(old_value)
            new_values.append(new_value)

            if view not in piece_view:
                if old_value != new_value:
                    #print("The piece on square", view, "has old value", old_value,
                    #      "and new value", new_value)
                    pass

        #print("End of the compare_gen_moves_eval function")
        #print("input (calculated) -> eb_eval: {0} ({1}), gm_eval {2} ({3})"
        #      .format(eb_eval, running_eval, gm_eval, new_eval))
        #print("The running eval was:", running_eval)
        print("The move was [{0}, {1}] ({2} by piece {3}, new/old phase = {4}/{5}"
              .format(start_sq, dest_sq, move_mod, base_board.look(start_sq),
                      new_phase, old_phase))
        #print("The piece view was", piece_view)
        #print("The base board was:")
        #bf.print_board(base_board, False)
        #print("The move board was:")
        #bf.print_board(move_board, False)
        return

    def check_quick_evaluation(self, cpp_board, white_to_play):
        """This function checks to see if generate_moves creates the same evaluations
        as eval_board"""

        gen_moves = bf.generate_moves(cpp_board, white_to_play)
        moves = gen_moves.get_moves()

        gm_eval = gen_moves.get_evaluation()
        eb_eval = bf.eval_board(cpp_board, white_to_play)

        if gm_eval != eb_eval:
            print("BASE: gm_eval was:", gm_eval, ", whilst eb_eval was:", eb_eval)
            base = False
        else: base = True

        all_same = True
        correct = 0

        for move in moves:

            gm_eval = move.get_evaluation()
            eb_eval = bf.eval_board(move.get_board(), not white_to_play)

            if gm_eval != eb_eval:
                all_same = False
                #print("gm_eval was:", gm_eval, ", whilst eb_eval was:", eb_eval,
                #      "the difference was", gm_eval - eb_eval)

                self.compare_gen_moves_eval(cpp_board, move, white_to_play)
                #return None, None, [None, None]
            else:
                correct += 1

        return all_same, base, [correct, len(moves)]

    def check_legal_moves(self, py_board, cpp_board, white_to_play):
        """This function checks to see if two boards have identical legal moves"""

         # find all the python legal moves
        legal_moves_py, data_array, evaluation, outcome = bl.total_legal_moves_plus(py_board, white_to_play);

        # find all the cpp legal moves
        tlm_struct = bf.total_legal_moves(cpp_board, white_to_play)
        legal_moves_cpp = tlm_struct.get_legal_moves()
        legal_moves_cpp = cpc.cpp_tlm_to_py(tlm_struct)

        # compare the output lists
        is_the_same = cpc.list_compare(legal_moves_py, legal_moves_cpp)

        if not is_the_same:
            print("white_to_play was", white_to_play)
            print("List one is python (correct)")
            print("List two is cpp")
            return False, None
        
        ## FOR TESTING
        #print("The two lists were identical")

        return True, legal_moves_py

    def identical_moves(self, py_board, cpp_board):
        """This function confirms two boards have exactly the same legal moves, and further
        that each move being made results in exactly the same board state"""

        # check that the incoming boards are identical
        is_identical = self.compare_board(py_board, cpp_board)
        if not is_identical:
            raise ValueError("the two input boards are not the same!")

        output_bool = True

        for white_to_play in [True, False]:

            good_tlm, legal_moves = self.check_legal_moves(py_board, cpp_board, white_to_play)

            if not good_tlm:
                output_bool = False
                continue

            # find all the python legal moves
            legal_moves_py,data_array,evaluation,outcome = bl.total_legal_moves_plus(py_board,white_to_play);

            # check that the checkmate functions agree
            mate_good = self.check_checkmate(py_board, cpp_board, white_to_play)

            # now we make every legal move on the board and check the result is always the same
            for i in range(len(legal_moves_py)):

                move_good, new_py, new_cpp = self.check_move(py_board, cpp_board, legal_moves_py[i])

                if not move_good:
                    output_bool = False
                    continue

                mate_good = self.check_checkmate(new_py, new_cpp, white_to_play)

                if not mate_good:
                    output_bool = False

        return output_bool

    def self_test(self):
        """This function self tests with all the moves in self.moves"""

        successful_tests = 0
        failed_tests = 0

        test_counter = 0

        for move_set in self.moves:
            test_counter += 1
            for i in range(len(move_set)):

                mini_set = move_set[:i+1]

                py_board, cpp_board = self.create_boards(mini_set)
                suceeded = self.identical_moves(py_board, cpp_board)

                if not suceeded:
                    print("Error! Not a perfect match in identical_moves")
                    failed_tests += 1
                    self.print_boards(py_board, cpp_board)

                    # FOR TESTING
                    print("Test counter ", test_counter)
                    print("i", i)
                    return
                else:
                    successful_tests += 1

        print("Testing complete")
        print("Successful board tests:", successful_tests)
        print("Failed board tests:", failed_tests)

        return

    def create_index_list(self, num):
        """This function creates an index list to loop over the boards saved
        in this class over and over until num is met"""

        num_boards = len(self.py_boards)

        if num_boards > num:
            full_loops = 0
            final_iter = num
        else:
            full_loops = num // num_boards
            final_iter = num % num_boards

        # construct a list of the indexes needed to fulfill the number of iterations
        index_list = []
        for i in range(full_loops):
            index_list += list(range(num_boards))
        index_list += list(range(final_iter))

        return index_list

    def timed_test(self, pad=True, tlm=True, eval=True, gen=True):
        """This function compares the time taken for python vs cpp functions"""

        pad_num = 15000         # no. of iterations for piece_attack_defend
        tlm_num = 500           # no. of iterations for total_legal_moves
        eval_num = 1000         # no. of iterations for eval_board
        gen_num = 4000          # no. of iterations for generate_moves
        white_to_play = True

        pad_indexes = self.create_index_list(pad_num)
        tlm_indexes = self.create_index_list(tlm_num)
        eval_indexes = self.create_index_list(eval_num)
        gen_indexes = self.create_index_list(gen_num)

        # test piece_attack_defend
        if pad:
            print("Running test on function: piece_attack_defend")
            # prepare variables
            square_num = 55
            our_piece = 5
            cpp_colour = 2 * white_to_play - 1
            our_colour = ((white_to_play - 1) * -2) + white_to_play
            t0_pad = time.process_time()
            for i in pad_indexes:
                pad_struct = bf.piece_attack_defend(self.cpp_boards[i], square_num, our_piece, cpp_colour)
            t1_pad = time.process_time()
            for j in pad_indexes:
                py_attack_list,py_attack_me,py_defend_me,py_piece_view = bl.piece_attack_defend(self.py_boards[i],
                                                          square_num, our_piece, our_colour)
            t2_pad = time.process_time()
            print("\tThe total number of iterations was", len(pad_indexes))
            print("\tThe time for cpp was: ",t1_pad - t0_pad)
            print("\tThe time for python was: ", t2_pad - t1_pad)
            print("\tcpp was faster by a factor of {0:.1f}".format((t2_pad - t1_pad) / (t1_pad - t0_pad + 1e-15)))

        # test total_legal_moves
        if tlm:
            print("Running test on function: total legal moves")
            t0_tlm = time.process_time()
            for i in tlm_indexes:
                tlm_struct = bf.total_legal_moves(self.cpp_boards[i], white_to_play)
                #discard = cpc.board_to_cpp(py_board)
                #cpp_eval = bf.eval_board(cpp_board, True)
            t1_tlm = time.process_time()

            for j in tlm_indexes:
                legal_moves_py,data_array,evaluation,outcome = bl.total_legal_moves_plus(self.py_boards[j], white_to_play)
                #py_eval = bl.eval_board_2(py_board, True);
            t2_tlm = time.process_time()
            print("\tThe total number of iterations was", len(tlm_indexes))
            print("\tThe time for cpp was: ", t1_tlm - t0_tlm)
            print("\tThe time for python was: ", t2_tlm - t1_tlm)
            print("\tcpp was faster by a factor of {0:.1f}".format((t2_tlm - t1_tlm) / (t1_tlm - t0_tlm + 1e-15)))

        # test eval_board
        if eval:
            print("Running test on function: eval_board")
            t0_eval = time.process_time()
            for i in eval_indexes:
                cpp_eval = bf.eval_board(self.cpp_boards[i], white_to_play)
            t1_eval = time.process_time()
            for j in eval_indexes:
                py_eval = bl.eval_board_2(self.py_boards[i], white_to_play)
            t2_eval = time.process_time()
            print("\tThe total number of iterations was", len(pad_indexes))
            print("\tThe time for cpp was: ",t1_eval - t0_eval)
            print("\tThe time for python was: ", t2_eval - t1_eval)
            print("\tcpp was faster by a factor of {0:.1f}".format((t2_eval - t1_eval) / (t1_eval - t0_eval + 1e-15)))

        # test generate_moves
        if gen:
            print("Running test on function: generate_moves")
            t0_gen = time.process_time()
            for i in gen_indexes:
                cpp_gen = bf.generate_moves(self.cpp_boards[i], white_to_play)
            t1_gen = time.process_time()
            print("\tThe total number of iterations was", len(gen_indexes))
            print("\tThe time for cpp was: ", t1_gen - t0_gen)
            print("\tThe average executions per second was: {0:.2f}".format(len(gen_indexes) / (t1_gen - t0_gen)))




