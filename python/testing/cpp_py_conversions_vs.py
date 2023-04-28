
# this module handles conversions from c++ to python and back
import board_func as bf
import board_light as bl

#from termcolour import colored as coloured

def board_to_py(board):
    '''This function converts a c++ board into a python board'''

    py_board = []

    for i in range(120):
        
        val = board.look(i)
        # if the square is out of bounds
        if abs(val) == 7:
            # note that en passant is currently ignored
            elem = [False, 0, 0, False]
            py_board.append(elem)
            continue
        # if the square is empty
        elif abs(val) == 0:
            elem = [True, 0, 0, False]
            py_board.append(elem)
            continue
        elif abs(val) == 1:
            elem = [True, 1, 0, False]
        elif abs(val) == 2:
            elem = [True, 2, 0, False]
        elif abs(val) == 3:
            elem = [True, 3, 0, False]
        elif abs(val) == 4:
            elem = [True, 4, 0, False]
        elif abs(val) == 5:
            elem = [True, 5, 0, False]
        elif abs(val) == 6:
            elem = [True, 6, 0, False]

        # now determine the colour
        if val < 0:
            elem[2] = 2
        else:
            elem[2] = 1

        py_board.append(elem)

    # now add the passant booleans
    for i in range(8):
        val = board.look(5+i)
        if val == 7:
            py_board[48-i][3] = True
    for j in range(8):
        val = board.look(13+j)
        if val == 7:
            py_board[78-j][3] = True
    # check the passant wipe boolean
    if board.look(4) == 7:
        py_board[4][3] = True
    # finally the castle rights booleans
    for k in range(4):
        val = board.look(k)
        if val == 7:
            py_board[k][3] = True
    # update the castle state
    if board.look(101) == -7 and board.look(102) == -7:
        py_board[5][3] = 0
    elif board.look(101) == 7 and board.look(102) == -7:
        py_board[5][3] = 1
    elif board.look(101) == -7 and board.look(102) == 7:
        py_board[5][3] = 2
    else:
        py_board[5][3] = 3
            
    # below are special info square for a pyboard, [4] and [100] are not set
    # # set the castle rights to all True
    # board[0][3] = 1    # white kingside castle rights
    # board[1][3] = 1    # white queenside castle rights
    # board[2][3] = 1    # black kingside castle rights
    # board[3][3] = 1    # black queenside castle rights
    # #board[4][3] = 0   # passant_wipe does not need to be set true
    # #board[5][3] = 0   # nobody has castled yet, 1=white only, 2=black only, 3=both
    # board[100][3] = 1    # phase 1 at the start
    

    return py_board

def board_to_cpp(py_board):
    '''This function converts a python board to cpp'''
    
    # create a cpp board object
    board = bf.create_board()
    
    # loop through the py_board and update the cpp board
    for i in range(120):
        
        val = py_board[i]
        
        if val[0] == False:
            board.set(i,-7)
            continue
            
        if val[2] == 1:
            sign = 1
        elif val[2] == 2:
            sign = -1
        else:
            sign = 0
            
        board.set(i,val[1]*sign)
        
    # finally, go through special behaviour, en passant booleans etc
    
    # now add the passant booleans
    for i in range(8):
        val = py_board[48-i][3]
        if val == True:
            board.set(5+i,7)
    for j in range(8):
        val = py_board[78-j][3]
        if val == True:
            board.set(13+j,7)
    # add the castle rights booleans
    for k in range(4):
        val = py_board[k][3]
        if val == True:
            board.set(k,7)
    # add the passant wipe boolean
    if py_board[4][3] == True:
        board.set(4,7)
    # set the castle state
    if py_board[5][3] == 1:
        board.set(101,7)
    elif py_board[5][3] == 2:
        board.set(102,7)
    elif py_board[5][3] == 3:
        board.set(101,7)
        board.set(102,7)

    return board
    
def cpp_pad_to_py(pad_struct):
    '''This function converts a piece_attack_defend structure from cpp to python'''
    
    attack_list_cpp = pad_struct.get_attack_list()
    attack_me_cpp = pad_struct.get_attack_me()
    defend_me_cpp = pad_struct.get_defend_me()
    piece_view = pad_struct.get_piece_view()
    
    attack_list = []
    attack_me = []
    defend_me = []
    
    for i in range(int(len(attack_list_cpp)/3)):
        ind = (i*3)
        new_elem = [attack_list_cpp[ind],
                    attack_list_cpp[ind+1],
                    attack_list_cpp[ind+2]]
        attack_list.append(new_elem)
        
    for j in range(int(len(attack_me_cpp)/3)):
        ind = (j*3)
        new_elem = [attack_me_cpp[ind],
                    attack_me_cpp[ind+1],
                    attack_me_cpp[ind+2]]
        attack_me.append(new_elem)
        
    for k in range(int(len(defend_me_cpp)/3)):
        ind = (k*3)
        new_elem = [defend_me_cpp[ind],
                    defend_me_cpp[ind+1],
                    defend_me_cpp[ind+2]]
        defend_me.append(new_elem)
        
    return attack_list, attack_me, defend_me, piece_view

def cpp_tlm_to_py(tlm_struct):
    '''This function converts the total_legal_moves data structure
    from cpp to python'''

    # we need the data array of board pad structs too
    legal_moves_cpp = tlm_struct.get_legal_moves()
    evaluation = tlm_struct.get_evaluation()
    outcome = tlm_struct.get_outcome()

    legal_moves = []
    
    for i in range(int(len(legal_moves_cpp)/3)):
        ind = (i*3)
        new_elem = [legal_moves_cpp[ind],
                    legal_moves_cpp[ind+1],
                    legal_moves_cpp[ind+2]]
        legal_moves.append(new_elem)

    # only one return to begin with, add the rest later
    return legal_moves

def cpp_moves_to_py(move_struct, best_num):
    '''This function converts a cpp generated move struct into a python
    friendly format'''

    # we want considerations list with elements [start_sq,dest_sq,new_eval,move_mod]
    # we want a board list with each element being a python board
    considerations = []
    board_list = []
    moves = move_struct.get_moves()

    # loop through all the moves in the position
    for i in range(move_struct.get_length()):

        # get the data for considerations
        start_sq = moves[i].get_start_sq()
        dest_sq = moves[i].get_dest_sq()
        move_mod = moves[i].get_move_mod()
        new_eval = moves[i].get_evaluation() / 1000.0

        new_elem = [start_sq, dest_sq, new_eval, move_mod]
        considerations.append(new_elem)
        
        # get only the best boards
        if i <= best_num:
            cpp_board = moves[i].get_board()
            py_board = board_to_py(cpp_board)
            board_list.append(py_board)

    return considerations, board_list, move_struct.is_mating_move()

def board_to_py(board):
    '''This function converts a c++ board into a python board'''

    py_board = []

    for i in range(120):
        
        val = board.look(i)
        # if the square is out of bounds
        if abs(val) == 7:
            # note that en passant is currently ignored
            elem = [False, 0, 0, False]
            py_board.append(elem)
            continue
        # if the square is empty
        elif abs(val) == 0:
            elem = [True, 0, 0, False]
            py_board.append(elem)
            continue
        elif abs(val) == 1:
            elem = [True, 1, 0, False]
        elif abs(val) == 2:
            elem = [True, 2, 0, False]
        elif abs(val) == 3:
            elem = [True, 3, 0, False]
        elif abs(val) == 4:
            elem = [True, 4, 0, False]
        elif abs(val) == 5:
            elem = [True, 5, 0, False]
        elif abs(val) == 6:
            elem = [True, 6, 0, False]

        # now determine the colour
        if val < 0:
            elem[2] = 2
        else:
            elem[2] = 1

        py_board.append(elem)

    # now add the passant booleans
    for i in range(8):
        val = board.look(5+i)
        if val == 7:
            py_board[48-i][3] = True
    for j in range(8):
        val = board.look(13+j)
        if val == 7:
            py_board[78-j][3] = True

    return py_board

def print_py_board_evals(board, white_to_play):
    '''This function goest through a python board, evaluating each piece and
    printing out these evaluations'''

    data_array = [ 0.0 for p in range(64) ]

    # determine the phase of the game, piece bonuses and resultant evaluation
    phase, bonus, phase_value = bl.determine_phase(board, white_to_play)
    
    #data_array[64][0] = phase_value
    running_eval = 0.0
    running_eval += phase_value
    print("The python phase value is", phase_value)

    piece_roundup = []
    
    # loop through every square in the board
    for i in range(8):
        for j in range(8):
            
            index = (2+i)*10 + (j+1)
            
            # if the square contains a piece
            if board[index][1] != 0:
                
                # extract information about the piece
                piece_type = board[index][1]
                piece_colour = board[index][2]
                
                # find out which squares it attacks (can move to) and which
                # friendly/opposing pieces are in view of it
                (attack_list,attack_me,
                defend_me,piece_view) = bl.piece_attack_defend(board,index,
                                                           piece_type,
                                                           piece_colour)
                
                # evaluate pieces from NOT white_to_play because this identifies
                # pieces that we have which are hanging
                (value,mate) = bl.eval_piece(board, index, white_to_play, bonus,
                                          piece_type, piece_colour, attack_list,
                                          attack_me, defend_me)
 
                # if there is a checkmate in the position
                if mate:
                    outcome = 1
                    if white_to_play:
                        value = -100
                    else:
                        value = 100
                    
                # save the information in the data array
                data_array[(i*8)+j] = value

                running_eval += value

                piece_roundup.append([index, piece_type, piece_colour, value])

    # now we have finished filling the data array

    print("The python running eval was",running_eval)

    layout = "{0:^4}{1:^6}{2:^6}{3:^6}{4:^6}{5:^6}{6:^6}{7:^6}{8:^6}"
    print(layout.format('','A','B','C','D','E','F','G','H'))

    piece_list = [''] * 8

    # now print the result
    for i in range(8):
        for j in range(8):
            
            index = (2+i)*10 + (j+1)

            piece_list[j] = str(round(data_array[(i*8)+j],1))

        # print the whole row
        print(layout.format(8-i,piece_list[0],
                            piece_list[1],
                            piece_list[2],
                            piece_list[3],
                            piece_list[4],
                            piece_list[5],
                            piece_list[6],
                            piece_list[7]))
    
    
    return piece_roundup

def print_cpp_board_evals(board, white_to_play):
    '''This function goest through a cpp board, evaluating each piece and
    printing out these evaluations'''

    data_array = [ 0.0 for p in range(64) ]

    # determine the phase on the board
    phase = bf.determine_phase(board, white_to_play);
    
    #data_array[64][0] = phase_value
    running_eval = phase.get_eval_adjust() / 1000.0
    print("The cpp phase value is", running_eval)

    piece_roundup = []
    
    # loop through every square in the board
    for i in range(8):
        for j in range(8):
            
            index = (2+i)*10 + (j+1)
            
            # if the square contains a piece
            if board.look(index) != 0:
                
                # extract information about the piece
                piece_type = board.look(index)

                if piece_type < 0:
                    piece_colour = -1
                    piece_type *= -1
                else:
                    piece_colour = 1
                

                # adjust to be cpp compatible
                if piece_colour == 2:
                    piece_colour = -1

                # analyse this piece
                pad_struct = bf.piece_attack_defend(board, index, piece_type, piece_colour);

                mate = False

                # evaluate the piece
                value = bf.eval_piece(board, white_to_play, index,
                    phase, piece_type, piece_colour, mate, pad_struct);
 
                # if there is a checkmate in the position
                if mate:
                    outcome = 1
                    if white_to_play:
                        value = -100
                    else:
                        value = 100
                    
                # save the information in the data array
                data_array[(i*8)+j] = value
                running_eval += value / 1000.0

                piece_roundup.append([index, piece_type, piece_colour, value])

    # now we have finished filling the data array

    print("The cpp running eval was", running_eval)

    layout = "{0:^4}{1:^6}{2:^6}{3:^6}{4:^6}{5:^6}{6:^6}{7:^6}{8:^6}"
    print(layout.format('','A','B','C','D','E','F','G','H'))

    piece_list = [''] * 8

    running_eval = 0

    # now print the result
    for i in range(8):
        for j in range(8):
            
            index = (2+i)*10 + (j+1)

            piece_list[j] = str(round(data_array[(i*8)+j]/1000.0,1))

        # print the whole row
        print(layout.format(8-i,piece_list[0],
                            piece_list[1],
                            piece_list[2],
                            piece_list[3],
                            piece_list[4],
                            piece_list[5],
                            piece_list[6],
                            piece_list[7]))
    

    return piece_roundup

def binary_lookup(item, dictionary):
    '''This method finds the index of item in a sorted dictionary of hash
    values. If the item is not present, it gives the index of the next item,
    so that dictionary.insert(index) will put it in the right place'''
            
    probe_start = 0
    probe_end = len(dictionary)
        
    # this function does not work if dictionary=[], instead use dictionary=[0]
    if probe_end == 0:
        raise ValueError('''Binary lookup cannot recieve an empty dictionary,
                            use [0] insead of empty dictionary []''')
        
    while True:
    
        # use a probe in the centre of the probe range (rounding down)
        probe = probe_start + (probe_end - probe_start)//2
            
        # have we finished?
        if probe == probe_start:
            # if item is the same as the probe
            if item == dictionary[probe]:
                present = True
                index = probe
            else:    # the item is not present
                present = False
                index = probe
            break
            
        # if item is the same as the probe
        if item == dictionary[probe]:
            present = True
            index = probe
            break
            
        # if the item is less than the probe
        elif item < dictionary[probe]:
            probe_end = probe
                
        # if the item is greater than the probe
        elif item > dictionary[probe]:
            probe_start = probe
        
    # index is the place to insert the new value, so +1 to make this work
    return present,index+1

def list_compare(list_one,list_two):
    '''This method checks whether two lists are exactly the same'''
        
    # first, convert each list to a hashlookup_list = [0]
    new_pair = [[], []]
    hash_list = [[0],[0]]
    old_pair = [list_one, list_two]
        
    for i in range(2):
        for item in old_pair[i]:
                
            # hash the item in the list
            hashed = hash(str(item))
                
            # sort these hashes into a new list
            present,index = binary_lookup(hashed,hash_list[i])
                
            # if the item is not already present
            if not present:
                    
                # save this non duplicate item in the new list
                new_pair[i].insert(index-1,item)
                    
                # save this non duplicate hash in the lookup list
                hash_list[i].insert(index,hashed)
                    
            else:    # we do not expect any duplicates
                
                print('Found a duplicate in the list')
                if i == 1:
                    print('The duplicate was on the new list')
                else:
                    print('The duplicate was on the old list')
                print('The entry was ',item)
                    
    wrong = 0
                    
    # now we have two pairs of sorted lists to directly compare
    for j in range(max(len(new_pair[0]),len(new_pair[1]))):
            
        # every element should be identical
        try:
            assert(new_pair[0][j] == new_pair[1][j])  
        except:    # in the case that they aren't
            wrong += 1
                
    is_the_same = True
                
    # now see whether the legal moves are the same
    if wrong > 0:
            
        is_the_same = False
        # mistakes were made
        # # print the board
        # bl.print_board(self.board_array_list[board_id[0]-self.offset]\
        #                 [board_id[1]],False)
        print('Length list one =',len(list_one),'Length list two = ',len(list_two))
        print('The number of errors was ',wrong)
        print('The sorted lists are:')
        print('One: ',new_pair[0])
        print('Two: ',new_pair[1])
        
    return is_the_same

def ordered_list(lst):
    """This function puts a list into a hash based order"""

    dictionary = [0]
    sorted_lst = []

    for item in lst:
        
        # hash the item in the list
        hashed = hash(str(item))

        # sort these hashes into a new list
        present, index = binary_lookup(hashed, dictionary)

        if not present:
            sorted_lst.insert(index - 1, item)
            dictionary.insert(index, hashed)
        else:
            print("Found a duplicate in the list!")

    return sorted_lst

def print_eval_comparison_board(cpp_pieces, py_pieces):
    """
    This function prints a board showing the differences between a cpp evaluation
    and a python evaluation
    """

    #piece_roundup.append([index, piece_type, piece_colour, value])

    print("Printing the difference between cpp pieces (+) and py (-) pieces:")

    piece_indexes = []
    for p in range(len(cpp_pieces)):
        piece_indexes.append(py_pieces[p][0])

    layout = "{0:^4}{1:^6}{2:^6}{3:^6}{4:^6}{5:^6}{6:^6}{7:^6}{8:^6}"
    print(layout.format('','A','B','C','D','E','F','G','H'))

    piece_list = [''] * 8

    # now print the result
    for i in range(8):
        for j in range(8):
            
            index = (2+i)*10 + (j+1)

            empty = True

            for k in range(len(piece_indexes)):
                if index == piece_indexes[k]:
                    empty = False
                    break

            if empty:
                piece_list[j] = '-'
            else:
                # get the difference in evaluations and piece colour
                if cpp_pieces[k][2] == 1:
                    piece_colour = 'w'
                    diff = (cpp_pieces[k][3]/1000) - py_pieces[k][3]
                elif cpp_pieces[k][2] == -1:
                    piece_colour = 'b'
                    diff = -(cpp_pieces[k][3]/1000) + py_pieces[k][3]
                
                # get the piece type
                if cpp_pieces[k][1] == 1: piece_type = 'P'
                elif cpp_pieces[k][1] == 2: piece_type = 'N'
                elif cpp_pieces[k][1] == 3: piece_type = 'B'
                elif cpp_pieces[k][1] == 4: piece_type = 'R'
                elif cpp_pieces[k][1] == 5: piece_type = 'Q'
                elif cpp_pieces[k][1] == 6: piece_type = 'K'

                if diff > 0:
                    piece_list[j] = '+' + str(round(diff, 2))
                else:
                    piece_list[j] = str(round(diff, 2))

                #piece_list[j] = (piece_colour + piece_type + "(" + 
                #                 str(round(diff, 2)) + ")")
                #piece_list[j] = str(round(diff, 2))

        # print the whole row
        print(layout.format(8-i,piece_list[0],
                            piece_list[1],
                            piece_list[2],
                            piece_list[3],
                            piece_list[4],
                            piece_list[5],
                            piece_list[6],
                            piece_list[7]))


def eval_piece_roundup(py_board, cpp_board, white_to_play):
    """This function compares the piece evaluations between a py and cpp board"""

    # print the base board state
    bf.print_board(cpp_board, False)

    # evaluate the board
    cpp_eval = bf.eval_board(cpp_board, white_to_play)
    print("The cpp evalution is {0:.2f} ( {1} )".format(cpp_eval/1000, cpp_eval))
    py_eval = bl.eval_board_2(py_board, white_to_play)
    print("The py evaluation is {0:.2f}".format(py_eval))
    print("-------------------------------------------")

    # print visualisation of evaluations
    py_pieces = print_py_board_evals(py_board, white_to_play)
    cpp_pieces = print_cpp_board_evals(cpp_board, white_to_play)
    print_eval_comparison_board(cpp_pieces, py_pieces)

    print("Now print more detailed text for each of the pieces:")

    ## order both lists (this doesn't work as the evaluations are different!)
    #py_pieces = ordered_list(py_pieces)
    #cpp_pieces = ordered_list(cpp_pieces)

    # confirm they are the same length
    assert(len(py_pieces) == len(cpp_pieces))

    wp = []
    wn = []
    wb = []
    wr = []
    wq = []
    wk = []
    bp = []
    bn = []
    bb = []
    br = []
    bq = []
    bk = []

    for i in range(len(py_pieces)):

        piece_square = py_pieces[i][0]
        piece_type = py_pieces[i][1]
        piece_colour = py_pieces[i][2]
        py_eval = py_pieces[i][3]
        cpp_eval = cpp_pieces[i][3] / 1000.0

        # check we are at the same entry
        assert(py_pieces[i][0] == cpp_pieces[i][0])
        assert(py_pieces[i][1] == cpp_pieces[i][1])
        if piece_colour == 1: assert(cpp_pieces[i][2] == 1)
        else: assert(cpp_pieces[i][2] == -1)

        if piece_type == 1: p = "pawn"
        elif piece_type == 2: p = "knight"
        elif piece_type == 3: p = "bishop"
        elif piece_type == 4: p = "rook"
        elif piece_type == 5: p = "queen"
        elif piece_type == 6: p = "king"

        if piece_colour == 1: c = "white"
        else: c = "black"

        d = cpp_eval - py_eval

        print_str = "The {0} {1} on square {2} has python value {3:.2f} and cpp value {4:.2f}, the difference is {5:.2f}".format(
            c, p, piece_square, py_eval, cpp_eval, d)

        if c == "white":
            if piece_type == 1: wp.append(print_str)
            elif piece_type == 2: wn.append(print_str)
            elif piece_type == 3: wb.append(print_str)
            elif piece_type == 4: wr.append(print_str)
            elif piece_type == 5: wq.append(print_str)
            elif piece_type == 6: wk.append(print_str)
        else:
            if piece_type == 1: bp.append(print_str)
            elif piece_type == 2: bn.append(print_str)
            elif piece_type == 3: bb.append(print_str)
            elif piece_type == 4: br.append(print_str)
            elif piece_type == 5: bq.append(print_str)
            elif piece_type == 6: bk.append(print_str)

        #print(print_str)

    # print the pieces out in groups
    j = 0
    for prnt in wp: j += 1; print(j,"\t", prnt);
    for prnt in wn: j += 1; print(j,"\t", prnt);
    for prnt in wb: j += 1; print(j,"\t", prnt);
    for prnt in wr: j += 1; print(j,"\t", prnt);
    for prnt in wq: j += 1; print(j,"\t", prnt);
    for prnt in wk: j += 1; print(j,"\t", prnt);
    for prnt in bp: j += 1; print(j,"\t", prnt);
    for prnt in bn: j += 1; print(j,"\t", prnt);
    for prnt in bb: j += 1; print(j,"\t", prnt);
    for prnt in br: j += 1; print(j,"\t", prnt);
    for prnt in bq: j += 1; print(j,"\t", prnt);
    for prnt in bk: j += 1; print(j,"\t", prnt);
        
        
    
    
    
    
    
    
