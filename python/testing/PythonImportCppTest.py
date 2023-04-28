
import board_func as bf
import board_light as bl
import cpp_py_conversions_vs as cpc
import sys
import chess_testing as ct
import StockfishCompare as SC

import engine_class as ec
eng = ec.chess_engine()

test_object = ct.TestChess(quick=False)
sc_comp = SC.StockfishCompare()

#test_object.self_test()
test_object.timed_test(pad=True, tlm=False, eval=False, gen=True)
#test_object.check_quick_evaluation(sc_comp.cpp_boards[4], True)

exit()

correct_boards = 0.0
correct_moves = 0.0
correct_base = 0.0
total_moves = 0.0
total_boards = 0.0

for i in range(len(test_object.cpp_boards)):
    b = test_object.cpp_boards[i]
    wtp = test_object.to_play[i]
    correct, base, [g, t] = test_object.check_quick_evaluation(b, wtp)
    # for testing
    if correct == None:
        break

    if correct:
        correct_boards += 1
    if base:
        correct_base += 1
    correct_moves += g
    total_moves += t
    total_boards += 1

print("Boards correct", correct_boards, "/", total_boards, 
      "({0:.1f}%)".format(100*correct_boards/total_boards))
print("Base evals correct", correct_base, "/", total_boards,
       "({0:.1f}%)".format(100*correct_base/total_boards))
print("Correct moves overall", correct_moves, "/", total_moves,
       "({0:.1f}%)".format(100*correct_moves/total_moves))

#test_object.check_quick_evaluation(test_object.cpp_boards[-1], True)

#test_object.check_evals(sc_comp.py_boards[4], sc_comp.cpp_boards[4], True)


#compare_object = SC.StockfishCompare()
#compare_object.compare(0, 0)

#for i in range(compare_object.num):

#    output = test_object.identical_moves(compare_object.py_boards[i], compare_object.cpp_boards[i])
#    print(output)





#-----------------------------------#

#py_board = bl.create_board(moves,True)

## make a super custom board
#py_board = bl.clear_board(py_board)
#py_board[31][1:3] = [1,1]
#py_board[32][1:3] = [1,1]
#py_board[33][1:3] = [1,1]

#py_board[73][1:3] = [1,2]
#py_board[74][1:3] = [1,2]
#py_board[75][1:3] = [1,2]

#py_board[41][1:3] = [1,1]
#py_board[76][1:3] = [1,1]
#py_board[61][1:3] = [6,2]
#py_board[24][1:3] = [6,1]
#py_board[87][1:3] = [1,2]



##py_board[82][1:3] = [6,2]
##py_board[42][1:3] = [1,2]
##py_board[54][1:3] = [3,1]
##py_board[84][1:3] = [2,1]
##py_board[74][1:3] = [1,2]
##py_board[75][1:3] = [5,2]
##py_board[52][1:3] = [5,1]
##py_board[53][1:3] = [2,1]
##py_board[61][1:3] = [1,1]
##py_board[93][1:3] = [2,2]
##py_board[64][1:3] = [2,2]
##py_board[72][1:3] = [1,2]
##py_board[81][1:3] = [3,2]
##py_board[41][1:3] = [6,1]
##py_board[32][1:3] = [1,1]
    
#print('The starting python board is')
#bl.print_board(py_board)

## now convert to cpp
#cpp_board = cpc.board_to_cpp(py_board)
#print('The cpp equivalent board is')
#bf.print_board(cpp_board, False)

#white_to_play = True

## evaluate the board
#cpp_eval = bf.eval_board(cpp_board, white_to_play)
#print("The cpp evaluation is ", cpp_eval, "( ", cpp_eval/1000, " )")
#py_eval = bl.eval_board_2(py_board, white_to_play)
#print("The py evaluation is ", py_eval)

## now do generate moves on the board
#gen_move_cpp = bf.generate_moves(cpp_board, white_to_play)
#considerations, board_list, mating_move = cpc.cpp_moves_to_py(gen_move_cpp, 5)

#print(considerations)
