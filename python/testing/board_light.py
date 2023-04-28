#import numpy as np
# COMMENTED OUT NUMPY FOR NOW, BEST TO FIX AFTER FINISHED TESTING C++

# Function determines if a move is legal
def is_legal(piece,colour,dest_sq,move_modifier):
    ''' return (True,move_modifier)'''
    '''Determines if a move is legal, under the assumption that '''
    '''the piece is able to move to that square eg for a bishop, it '''
    '''assumes the piece is on that bishops diagonals. This does not'''
    '''check for castling or checks'''

    # if the square is out of bounds
    if dest_sq[0] == 0:
        return (False,None)

    # does the square contain a piece
    if dest_sq[1] != 0:
        
        # if the square contains a piece of the same colour
        if dest_sq[2] == colour:
            return (False,None)
        else:    # if the square contains an opposition piece
            # if the piece is unable to capture
            if move_modifier != 2:
                return (False,None)
            else:    # the piece can capture
                
                # is the piece the king
                if dest_sq[1] == 6:
                    move_modifier = 4

    else:    # the square is empty
        
        # check if the piece is a pawn for special behaviour
        if piece == 1:
        
            # if the pawn is trying to capture
            if move_modifier == 2:
                
                # check if en passant is being requested
                if dest_sq[3] == True:
                    move_modifier = 3
                    return (True,move_modifier)
                else:    # it is illegal for a pawn to move diagonally
                    return(False,None)
            
        else:    # if the piece is not a pawn, downgrade from capture to move
                 # it is important not to do this for pawns that are promoting
            move_modifier = 1
            

    # so, the square is either empty or contains a capturable piece

    # no illegal stuff going on
    return (True,move_modifier)

#-----------------------------------------------------------------------------

def piece_moves(piece_type,piece_colour,square):
    '''This function returns the move_values and move_depth for a piece, '''
    ''' these are where and how a piece can move on the board'''
    
    # go through each type of piece and assign values
    if piece_type == 0:
        return ([],None)
    
    elif piece_type == 1:
        if piece_colour == 2:
            move_values = [[-9,2],
                           [-10,1],
                           [-11,2]]
            
            # if the black pawn is on the 7th rank
            if square // 10 == 8:
                move_depth = 2
            else:
                move_depth = 1
                
        elif piece_colour == 1:
            move_values = [[9,2],
                           [10,1],
                           [11,2]]
            
            # if the white pawn is on the 2nd rank
            if square // 10 == 3:
                move_depth = 2
            else:
                move_depth = 1
                
        else:
            print('Player does not have any colour')
            
    elif piece_type == 2:
        
        move_values = [[-21,2],
                       [-19,2],
                       [-12,2],
                       [-8,2],
                       [8,2],
                       [12,2],
                       [19,2],
                       [21,2]]
        move_depth = 1
    
    elif piece_type == 3:
        
        move_values = [[-11,2],
                       [-9,2],
                       [9,2],
                       [11,2]]
        move_depth = 7
        
    elif piece_type == 4:
        
        move_values = [[-10,2],
                       [-1,2],
                       [1,2],
                       [10,2]]
        move_depth = 7
        
    elif piece_type in [5,6]:

        move_values = [[-11,2],
                       [-10,2],
                       [-9,2],
                       [-1,2],
                       [1,2],
                       [9,2],
                       [10,2],
                       [11,2]]
        
        if piece_type == 6:
            move_depth = 1
        else:
            move_depth = 7
            
    return (move_values,move_depth)
            
    # now the piece move values and modifiers are set
    # end of piece_moves function
    
#-----------------------------------------------------------------------------#

def is_in_check(board,king_sq_no,king_colour,boolean=False):
    '''return list_of_checks (or if boolean=True return True/False'''
    '''This function checks if the king on king_sq is in check'''
    
    # if the specified square is out of bounds
    if board[king_sq_no][0] == False:
        return None
    
    # groups of pieces that have the same movements
    move_groups = [[6,1],
                   [2,2],
                   [3,5],
                   [4,5]]
    
    list_of_checks = []
    
    # for each of the four groups of pieces
    for i in range(4):
        
        piece_types = move_groups[i]
        
        # find out how this group of pieces moves
        (move_values,move_depth) = piece_moves(piece_types[0],1,55)
        
        # check all the squares that these pieces can check the king from
        for (move_dist,move_modifier) in move_values:
            
            # arithmatic begins at kings square
            dest_sq = king_sq_no
            
            # check all the squares these pieces can reach
            for j in range(move_depth):
            
                dest_sq = dest_sq + move_dist
                
                # is the square out of bounds
                if board[dest_sq][0] == False:
                    break
                
                # if the square contains an opposing piece of this type
                if (board[dest_sq][1] in piece_types
                    and board[dest_sq][2] != king_colour):
                    
                    # check if the attacking piece is a pawn
                    if board[dest_sq][1] == 1:
                        if king_colour == 1:
                            if dest_sq not in [king_sq_no+9,king_sq_no+11]:
                                continue # because the pawn cannot check
                        else:  # the king is black
                            if dest_sq not in [king_sq_no-9,king_sq_no-11]:
                                continue # because the pawn cannot check
                    
                    # hence at this point the king is in check
                    if boolean == True:
                        return True
                    else:
                        list_of_checks.append([dest_sq,board[dest_sq][1]])
                    
                    break # since no more checks can occur along this line
                
                # if the square contains a blocking piece
                elif board[dest_sq][1] != 0:
                    break # since no more checks can occur along this line
    
    # if function is set to boolean mode then there have been no checks
    if boolean:
        return False
    
    # else all the checks are saved in the list_of_checks
    return list_of_checks

#-----------------------------------------------------------------------------#

def can_castle(board,piece_colour,boolean=False):
    '''returns list_of_castles (or return True/False if boolean = True)'''
    '''This function determines if the player colour specified can castle'''
    
    # check if the player has any castle rights
    if piece_colour == 1:
        i = 0 # indexing variable
        if not (board[0][3] or board[1][3]):
            if boolean:
                return False
            else:
                return []
    else:
        i = 2 # indexing variable
        if not (board[2][3] or board[3][3]):
            if boolean:
                return False
            else:
                return []
    
    # define some castling information
    list_of_castles = []
    move_modifiers = [[22,5],[26,5],
                      [92,5],[96,5]]
    squares_no_check = [[22,23,24],[24,25,26],
                          [92,93,94],[94,95,96]]
    squares_needed_empty = [[22,23,23],[25,26,27],[92,93,93],[95,96,97]]
            
    # loop through kingside and queenside
    for j in range(2):
        
        # if the player has the right to castle on this side
        if board[i+j][3]:
            
            # check if squares are empty
            if (board[squares_needed_empty[i+j][0]][1] == 0 and
                board[squares_needed_empty[i+j][1]][1] == 0 and
                board[squares_needed_empty[i+j][2]][1] == 0):
                
                # check if any of the three castling squares are in check
                if not (is_in_check(board,squares_no_check[i+j][0],piece_colour,True) or
                    is_in_check(board,squares_no_check[i+j][1],piece_colour,True) or
                    is_in_check(board,squares_no_check[i+j][2],piece_colour,True)):
                    
                    # save the castling move
                    list_of_castles.append(move_modifiers[i+j])
                    
                    if boolean:
                        return True
                    
    # if in boolean mode, return False if no castling is possible
    if boolean and list_of_castles == []:
        return False
    else:
        return list_of_castles
            
#-----------------------------------------------------------------------------#

def find_check(board,king_colour,boolean=True):
    '''This function finds out if the king is in check in a board configuration'''
    for i in range(8):
        for j in range(8):
            index = (2+i)*10  +  (j+1)
            if board[index][1] == 6 and board[index][2] == king_colour:
                return is_in_check(board,index,king_colour,boolean)
            
#-----------------------------------------------------------------------------#

def move_piece_on_board(board,start_sq,dest_sq,pawn_start=False,passant_wipe=False):
    '''This function actually moves the piece, wiping over the information in 
    the target square and switching the passant booleans'''
    
    board[dest_sq][1:3] = board[start_sq][1:3]
    board[start_sq][1:3] = [0,0]
    
    # if indicated, all en passant booleans are reset to False
    if passant_wipe == True:
        for i in [41,71]:
            for j in range(8):
                board[i+j][3] = False
        
        # now that the passant booleans have been , passant wipe = 0
        board[4][3] = False
    
    # if this was a pawns first move, set en passant possible
    if pawn_start == True:
        if board[dest_sq][2] == 1:
            board[start_sq+10][3] = True
        else:
            board[start_sq-10][3] = True
        
        # save the fact that this passant boolean needs to be wiped later
        # ie passant wipe = 1
        board[4][3] = True
            
    return board

#-----------------------------------------------------------------------------#



#-----------------------------------------------------------------------------#
    
def create_board(move_order=[],string_input=False):
    '''Initialise the pieces to the starting position, and execute a '''
    '''move order if one is supplied'''
    
    n = 0
    
    board = [[0,0,0,0] * 1 for i in range(120)]

    for row in range(12):
        for clm in range(10):
            
            # set board_y_n to True if it is a playable square
            if (row == 0) or (row == 1) or (row == 10) or (row == 11):
                board[n][0] = False
            elif (clm == 0) or (clm == 9):
                board[n][0] = False
            else:
                board[n][0] = True
                
            # setup the pieces
            if (row in [2,9]) and (clm in [1,8]):
                board[n][1] = 4
            elif (row in [2,9]) and (clm in [2,7]):
                board[n][1] = 2
            elif (row in [2,9]) and (clm in [3,6]):
                board[n][1] = 3
            elif (row in [2,9]) and (clm == 4):
                board[n][1] = 6
            elif (row in [2,9]) and (clm == 5):
                board[n][1] = 5
            elif (row in [3,8]) and (clm in [1,2,3,4,5,6,7,8]):
                board[n][1] = 1
            else:
                board[n][1] = 0
                
            # setup the colours of the pieces
            if (row in [2,3]) and (clm in [1,2,3,4,5,6,7,8]):
                board[n][2] = 1
            elif (row in [8,9]) and (clm in [1,2,3,4,5,6,7,8]):
                board[n][2] = 2
            else:
                board[n][2] = 0
                
            # set that en passant is not possible for any square
            board[n][3] = 0
                
            # increment the index
            n += 1
            
    # set the castle rights to all True
    board[0][3] = 1    # white kingside castle rights
    board[1][3] = 1    # white queenside castle rights
    board[2][3] = 1    # black kingside castle rights
    board[3][3] = 1    # black queenside castle rights
    #board[4][3] = 0   # passant_wipe does not need to be set true
    #board[5][3] = 0   # nobody has castled yet
    board[100][3] = 1    # phase 1 at the start
            
    #now if a move order is supplied, apply it to the board
    white_to_move = True
    
    for move in move_order:
        (is_legal,board_index,move_modifier) = interpret_move(board,move,
                                                            white_to_move,
                                                            string_input)
        if not is_legal:
            print('Error in the supplied move order, the following is illegal: ',
                  ind_to_letter(move))
            break
        
        make_move(board,board_index,move_modifier)
        
        # now it is the other persons turn
        white_to_move = not white_to_move
        
                
    return board
        

# now more...

#-----------------------------------------------------------------------------#

def legal_moves(board,square):
    '''return list_of_legal_moves'''
    '''Creates a list of legal moves from a square'''   
    
    # extract information about the square
    piece_type = board[square][1]
    piece_colour = board[square][2]
    
    # determine how the piece moves, if there is one
    (move_values,move_depth) = piece_moves(piece_type,piece_colour,square)
    
    list_of_legal_moves = []          # initialise variable
        
    # for all of the possible ways of moving the piece
    for (move_dist,move_modifier) in move_values:
        
        # arithmatic begins at starting square
        dest_sq = square
        
        # loop for each square the piece can move in a row
        for i in range(move_depth):
            
            dest_sq = dest_sq + move_dist
            (move_is_legal,move_type) = is_legal(piece_type,
                                                piece_colour,
                                                board[dest_sq],
                                                move_modifier)
            
            # either save the move or break from the loop if its illegal
            if move_is_legal == False:
                break
            else:
                list_of_legal_moves.append([dest_sq,move_type])
                
                # if the move was a capture
                if move_type in [2,4]:
                    break
                
    
        
    # see if any of the moves lead to check, and remove if so
    for (dest_sq,move_type) in list_of_legal_moves[:]:
        
        # copy the entire board
        board_state = [x[:] for x in board]
        
        # make the move on the copied board
        board_state = move_piece_on_board(board_state,square,dest_sq)
        
        # if the resultant position has a check in it to our king
        if find_check(board_state,piece_colour):
            
            # remove the move from the list
            list_of_legal_moves.remove([dest_sq,move_type])
                          
    # check if the piece is a king
    if piece_type == 6:
        # see if it can legally castle
        list_of_castles = can_castle(board,piece_colour)
        if list_of_castles != []:
            for castling in list_of_castles:
                list_of_legal_moves.append(castling)
    
    # if a pawn reaches rank 1 or 8, it must be promoting
    if piece_type == 1:
        for i in range(len(list_of_legal_moves)):
            # if the king is in check, don't set move type to promote
            if list_of_legal_moves[i][0]//10 in [2,9] and list_of_legal_moves[i][1] != 4:
                # change this entry to knight promotion
                list_of_legal_moves[i][1] = 6
                # add queen promotion to the list
                list_of_legal_moves.append([list_of_legal_moves[i][0],7])
                # add bishop promotion to the list
                list_of_legal_moves.append([list_of_legal_moves[i][0],8])
                # add rook promotion to the list
                list_of_legal_moves.append([list_of_legal_moves[i][0],9])
    
    # now there is a list of legal moves that can be made
    return list_of_legal_moves

#-----------------------------------------------------------------------------#

def print_board(board,tidy=True):
    '''This method prints the board in a crude fashion'''
    
    # if tidy is set to true, print the board without numerical references,
    # only chess references eg A,B,C... 1,2,3....
    if tidy:
        layout = "{0:^4}{1:^3}{2:^3}{3:^3}{4:^3}{5:^3}{6:^3}{7:^3}{8:^3}"
        print(layout.format('','A','B','C','D','E','F','G','H'))
    else: # print with chess references and numerical references to squares
        layout = "{0:<6}{1:>2}{2:^6}{3:^6}{4:^6}{5:^6}{6:^6}{7:^6}{8:^6}{9:^6}"
        print(layout.format('','','H(1)','G(2)','F(3)','E(4)','D(5)','C(6)','B(7)','A(8)'))
    
    piece_list = [''] * 8
    
    for i in range(8):
        for j in range(8):
            
            if tidy:
                index = (9-i)*10 + (8-j)
            else:
                index = (i+2)*10  + (j+1)
            
            if board[index][1] == 0:
                s1 = '.'
            else:
                if board[index][2] == 1:
                    s1 = 'w'
                else:
                    s1 = 'b'
            
            if board[index][1] == 0:
                s2 = ''
            elif board[index][1] == 1:
                s2 = 'P'
            elif board[index][1] == 2:
                s2 = 'N'
            elif board[index][1] == 3:
                s2 = 'B'
            elif board[index][1] == 4:
                s2 = 'R'
            elif board[index][1] == 5:
                s2 = 'Q'
            elif board[index][1] == 6:
                s2 = 'K'
                
                
            piece_list[j] = s1 + s2
            
        if tidy:
            print(layout.format(8-i,piece_list[0],
                            piece_list[1],
                            piece_list[2],
                            piece_list[3],
                            piece_list[4],
                            piece_list[5],
                            piece_list[6],
                            piece_list[7]))
        else:
            print(layout.format((i+2)*10,i+1,piece_list[0],
                            piece_list[1],
                            piece_list[2],
                            piece_list[3],
                            piece_list[4],
                            piece_list[5],
                            piece_list[6],
                            piece_list[7]))
        
#-----------------------------------------------------------------------------#
def clear_board(board):
    '''This method clears all pieces off the board'''
    for i in range(8):
        for j in range(8):
            index = (i+2)*10  +  (j+1)
            board[index][1:3] = [0,0]
    return board
    
#-----------------------------------------------------------------------------#
   
def interpret_move(board,move_command,white_to_move,string_input=True):
    '''This function interprets a move command eg e2e4 and converts it into '''
    '''a start square and destination square eg 35->55'''
    
    is_legal = False
    
    if move_command == []:
        return (is_legal, [], [])
    
    move_modifier = 'None'
    
    # if the move provided is in a string format eg e2e4
    if string_input:
        
        board_index = [0,0]
    
        start_file = move_command[0].lower()
        start_rank = move_command[1]
        end_file = move_command[2].lower()
        end_rank = move_command[3]
        
        board_index = [0,0]
        move_modifier = 'None'
        
        # check that the move input makes sense
        if (start_file not in ['a','b','c','d','e','f','g','h'] or
            end_file not in ['a','b','c','d','e','f','g','h'] or
            start_rank not in ['1','2','3','4','5','6','7','8'] or
            end_rank not in ['1','2','3','4','5','6','7','8']):
            print('Incorrect move entry')
            return (is_legal,board_index,move_modifier)
        
        letter_inputs = [start_file, end_file]
        number_inputs = [start_rank, end_rank]
        
        # for both starting and ending squares, find out their board index
        for i in range(2):
        
            row = 0
            clm = 0
            
            # convert the input into a start and end square
            for letter in ['h','g','f','e','d','c','b','a']:
                if letter == letter_inputs[i]:
                    break
                clm += 1
                
            for number in ['1','2','3','4','5','6','7','8']:
                if number == number_inputs[i]:
                    break
                row += 1
                
            board_index[i] = (2+row)*10  + (1+clm)
        
    else:
        board_index = move_command
        
    # confirm that the piece being moved belongs to the correct player
    if white_to_move:
        if board[board_index[0]][2] != 1:
            print('''It is white's move silly!''')
            return (is_legal,board_index,move_modifier)
    else:
        if board[board_index[0]][2] != 2:
            print('''It is black's move silly!''')
            return (is_legal,board_index,move_modifier)

    # now check what moves are legal from the starting square
    list_of_moves = legal_moves(board,board_index[0])
    
    # check if any of the moves are a promotion
    for (dest_sq,move_modifier) in list_of_moves:
        if dest_sq == board_index[1]:
            is_legal = True
            break
        
    if is_legal == False:
        print('Illegal move entered')
        ind = 0
        
        # convert the list of moves to a readable format
        for (board_num,move_type) in list_of_moves[:]:

            new_str = []
            board_str = str(board_num)
            if   board_str[1] == '1':  new_str = 'H'   
            elif board_str[1] == '2':  new_str = 'G'
            elif board_str[1] == '3':  new_str = 'F'
            elif board_str[1] == '4':  new_str = 'E'
            elif board_str[1] == '5':  new_str = 'D'
            elif board_str[1] == '6':  new_str = 'C'
            elif board_str[1] == '7':  new_str = 'B'
            elif board_str[1] == '8':  new_str = 'A'
            if   board_str[0] == '2':  new_str += '1'
            elif board_str[0] == '3':  new_str += '2'
            elif board_str[0] == '4':  new_str += '3'
            elif board_str[0] == '5':  new_str += '4'
            elif board_str[0] == '6':  new_str += '5'
            elif board_str[0] == '7':  new_str += '6'
            elif board_str[0] == '8':  new_str += '7'
            elif board_str[0] == '9':  new_str += '8'
            
            list_of_moves[ind][0] = new_str
            ind += 1

        print('The piece you selected can make the following legal moves: \n',list_of_moves)
        
    # finally, check if the move is a promotion
    if move_modifier in [6,7,8,9]:
        # have we been told what piece to promote to?
        if string_input and len(move_command) == 5:
            if move_command[4].lower() == 'n' or move_command[4].lower() == 'k':
                move_modifier = 6
            elif move_command[4].lower() == 'b':
                move_modifier = 8
            elif move_command[4].lower() == 'r':
                move_modifier = 9
            elif move_command[4].lower() == 'q':
                move_modifier = 7
            else:
                move_modifier = 7
        elif not string_input and len(move_command) == 3:
            move_modifier == move_command[2]
            if move_modifier not in [6,7,8,9]:
                move_modifier = 7
        else:    # we have not been told
            # the default is promotion to a queen
            move_modifier = 7
        
    return (is_legal,board_index,move_modifier)
        

#-----------------------------------------------------------------------------#

def move_piece(board,start_sq,dest_sq):
    '''This function prepares to move a piece on the board, with the final
    move being done in move_piece_on_board'''
    
    # check if its a pawn move that could allow en passant capture
    if ((start_sq // 10 in [3,8]) and (dest_sq // 10 in [5,6]) and
        board[start_sq][1] == 1):
        pawn_start = True
    else:
        pawn_start = False
        
    # check if this move removes castle rights
    if (board[start_sq][1] in [6,4] and
         (board[0][3] or board[1][3] or
         board[2][3] or board[3][3])):
        
        if board[start_sq][2] == 2:
            ind = 2
        else:
            ind = 0
        
        if board[start_sq][1] == 6:
            num = 2
        else:
            num = 1
            if start_sq % 10 == 1:
                pass
            elif start_sq % 10 == 8:
                ind = ind+1
            else:
                num = 0
        
        # save the removal of castle rights
        for i in range(num):
            board[ind+i][3] = False
    
    # note that board[4][3] is the passant wipe variable, either 1 or 0
    move_piece_on_board(board,start_sq,dest_sq,pawn_start,board[4][3])
    
        
#-----------------------------------------------------------------------------#
    
def any_check(board):
    '''This function determines if there are any checks on the board'''
    if find_check(board,1):
        return True
    elif find_check(board,2):
        return True
    else:
        return False
    
#-----------------------------------------------------------------------------#
    
def make_move(board,board_index,move_modifier):
    '''This method makes a move on the board in its entirety, taking '''
    '''account of behaviour such as castling and capture en passant'''
    
    # make the move on the board
    move_piece(board,board_index[0],board_index[1])
    
    # save the move for reference
    #move_order.append(board_index)
    
    # if the move was to an empty square
    if move_modifier == 1:
        pass
    
    # if the move was a capture
    elif move_modifier == 2:
        pass
    
    # if a pawn has captured en passant
    elif move_modifier == 3:
        # determine which pawn has been captured
        if board[board_index[1]][2] == 1:
            board[board_index[1]-10][1:3] = [0,0]
        else:
            board[board_index[1]+10][1:3] = [0,0]

    # if the move was a castle
    elif move_modifier == 5:
        # move the rook as well and update the castle information
        if board_index[1] == 22:
            move_piece(board,21,23)
            # if black has castled, set to 3: both have castled
            if board[5][3] == 2:
                board[5][3] = 3
            else:    # set to 1: white has castled
                board[5][3] = 1
        elif board_index[1] == 26:
            move_piece(board,28,25)
            # if black has castled, set to 3: both have castled
            if board[5][3] == 2:
                board[5][3] = 3
            else:    # set to 1: white has castled
                board[5][3] = 1
        elif board_index[1] == 92:
            move_piece(board,91,93)
            # if white has castled, set to 3: both have castled
            if board[5][3] == 1:
                board[5][3] = 3
            else:    # set to 2: black has castled
                board[5][3] = 2
        elif board_index[1] == 96:
            move_piece(board,98,95)
            # if white has castled, set to 3: both have castled
            if board[5][3] == 1:
                board[5][3] = 3
            else:    # set to 2: black has castled
                board[5][3] = 2
                
    # if the piece is promoting to a knight
    elif move_modifier == 6:
        board[board_index[1]][1] = 2
    
    # if the piece is promoting to a queen
    elif move_modifier == 7:
        board[board_index[1]][1] = 5
        
    # if the piece is promoting to a bishop (human only)
    elif move_modifier == 8:
        board[board_index[1]][1] = 3
        
    # if the piece is promoting to a rook (human only)
    elif move_modifier == 9:
        board[board_index[1]][1] = 4
    
    # # if the piece is promoting
    # if move_modifier == 6:
    #     please_promote = True
    #     while please_promote:
    #         # FIX THIS LATER PLEASE ALL PROMOTIONS ARE QUEEN FOR NOW
    #         #promo_piece = input('What would you like to promote to? Q/R/B/N  >   ')
    #         promo_piece = 'q'
    #         please_promote = False
    #         # logic to replace pawn with promoted piece
    #         if promo_piece.lower() == 'q':
    #             board[board_index[1]][1] = 5
    #         elif promo_piece.lower() == 'r':
    #             board[board_index[1]][1] = 4
    #         elif promo_piece.lower() == 'b':
    #             board[board_index[1]][1] = 3
    #         elif promo_piece.lower() in ['k','n']:
    #             board[board_index[1]][1] = 2
    #         else:
    #             print('You did not select a valid option')
    #             please_promote = False
            
#         # see if this move resulted in check
#         if any_check: # THIS IS NOT WORKING CORRECTLY
#             print('+')
        
#-----------------------------------------------------------------------------#

def piece_attack_defend(board,square_num,our_piece,our_colour):
    '''This function evaluates the value of a square to one side or the
    other'''
    
    if our_piece == 0:
        print('Empty square given to piece_attack_defend')
        return None
    
    # groups of pieces that have the same movements
    move_groups = [[6,1],
                   [2,2],
                   [3,5],
                   [4,5]]
    
    # this function will fill up these four lists
    attack_list = []      # squares this piece attacks
    attack_me = []        # pieces that attack this piece
    defend_me = []        # pieces that defend this piece
    piece_view = []       # pieces in view of this square (all of above)

    # are we a pawn?
    if our_piece == 1:
        pawn = True
    else:
        pawn = False
      
    # squares from which we could be attacked en passant
    en_passant = [square_num-1,square_num+1]
        
    if our_colour == 1:
        pawn_attacks = [square_num+9,square_num+11]
        pawn_defends = [square_num-9,square_num-11]
        passant_sq = -10
    else:
        pawn_attacks = [square_num-9,square_num-11]
        pawn_defends = [square_num+9,square_num+11]
        passant_sq = 10
        
    
    # for each of the four groups of pieces
    for i in range(4):
        
        piece_types = move_groups[i]
        
        # find out how this group of pieces moves
        (move_values,move_depth) = piece_moves(piece_types[0],1,55)
        
        # save if we are on the king/pawn loop
        if i == 0:
            first_loop = True
        else:
            first_loop = False
        
        # check all the squares that these pieces can move along
        for (move_dist,move_modifier) in move_values:
            
            # arithmatic begins at starting square
            dest_sq = square_num
            
            # set booleans along each line
            hostile_block = False
            friendly_block = False
            second_block = False
            attack = -1
            defend = 1
            priority = -1
            
            friendly_cover = False
            hostile_cover = False
            
            # check all the squares these pieces can reach
            for j in range(move_depth):
                
                # move to a new square each loop
                dest_sq += move_dist
                
                # is this square legal?
                if board[dest_sq][0] == False:
                    break
                
                # does the square contain a piece
                if board[dest_sq][1] != 0:
                    
                    # save the view of the piece (which pieces it can see)
                    piece_view.append(dest_sq)
                    
                    # if yes, check the colour of the piece
                    if board[dest_sq][2] == our_colour:
                        
                        # if its the same colour, check if it defends us
                        if board[dest_sq][1] in piece_types:
                            
                            # check if its a pawn, in which case it may not defend us
                            if first_loop and board[dest_sq][1] == 1:
                                if dest_sq not in pawn_defends:
                                    continue # because the pawn is not defending
                                        
                            # so the piece in the square defends us
                            
                            # define the new element for the defend list
                            new_elem = [dest_sq,board[dest_sq][1],defend]
                            
                            # check if the sight of this piece is blocked
                            if friendly_block:
                                new_elem[2] = 20+block_piece
                                defend_me.append(new_elem)
                            elif hostile_block:
                                new_elem[2] = 30+block_piece
                                defend_me.append(new_elem)
                            else:
                                # defend_list.append(new_elem)
                                defend_me.append(new_elem)
                                defend += 1
                                
                                # we are behind a friendly piece along this line
                                priority -= 1
                                friendly_cover = True
                                block_piece = board[dest_sq][1]
                            
                        else:    # its our colour but it doesn't defend us
                        
                            # this piece will be picked up in another loop
                            # no need to save any information
                            
                            # CODE THAT IGNORES BLOCKING PAWNS FOR BISHOPS/QUEEN DIAG ATTACKS
                            # if we are looking diagonally, and this piece is a
                            # pawn, it may already defend us and have pieces
                            # behind it which also directly defend us
                            if (board[dest_sq][1] == 1 and piece_types[0] == 3
                                and [dest_sq,1,1] in defend_me):
                                
                                # the pawn does not block along this line
                                defend +=1
                                
                                friendly_cover = True
                                block_piece = 1
                                continue
                            
                            # if we have already passed a blocking pawn and come
                            # to another piece that blocks the line, this is
                            # essentially the second block
                            if friendly_cover or hostile_cover:
                                second_block = True
                            # END OF CODE THAT IGNORES BLOCKING PAWNS
                            
                            # hence the piece is blocking
                            
                            if second_block:
                                break
                        
                            # this piece blocks along this line
                            friendly_block = True
                            second_block = True
                            block_piece = board[dest_sq][1]
                            defend = 1
                            
                    else:    # the square contains an opposing piece
                    
                        # check if the piece is attacking us
                        if board[dest_sq][1] in piece_types:
                            
                            # check if its a pawn, in which case it may not attack us
                            if first_loop and board[dest_sq][1] == 1:
                                if dest_sq not in pawn_attacks: 
                                    # unless it can capture en passant, continue
                                    if (board[square_num+passant_sq][3] == False
                                        or dest_sq not in en_passant):
                                        # so the pawn cannot attack us, but can
                                        # we attack the pawn?
                                        if (our_piece in piece_types and not
                                            (pawn and dest_sq not in pawn_attacks)):
                                            attack_list.append([dest_sq,board[dest_sq][1],priority])
                                        continue   # the pawn cannot attack us
                            
                            # so the piece in the square attacks us
                            
                            # if we have already passed a friendly blocking piece
                            if friendly_cover:
                                friendly_block = True
                                second_block = True
                                defend = 1
                            
                            # define the new element for the defend list
                            new_elem = [dest_sq,board[dest_sq][1],attack]
                            
                            # check if the sight of this piece is blocked
                            if friendly_block:
                                new_elem[2] = -20-block_piece
                                attack_me.append(new_elem)
                                # check if we can attack the piece
                                if (our_piece in piece_types and not
                                    (pawn and dest_sq not in pawn_attacks)):
                                    attack_list.append(new_elem)
                            elif hostile_block:
                                new_elem[2] = -30-block_piece
                                attack_me.append(new_elem)
                                # check if we can attack the piece
                                if (our_piece in piece_types and not
                                    (pawn and dest_sq not in pawn_attacks)):
                                    attack_list.append(new_elem)
                            else:    # the sight of the piece is not blocked
                                attack_me.append(new_elem)
                                attack -= 1
                                hostile_cover = True
                                block_piece = board[dest_sq][1]
                                
                                # check if we can attack the piece
                                if (our_piece in piece_types and not
                                    (pawn and dest_sq not in pawn_attacks)):
                                    attack_list.append([dest_sq,board[dest_sq][1],priority])
                                    
                                    priority -= 1
                            
                        else:    # its an opposing piece but it doesn't attack us
                            
                            # CODE THAT IGNORES BLOCKING PAWNS FOR BISHOPS/QUEEN DIAG ATTACKS
                            # if we are looking diagonally, and this piece is a
                            # pawn, it may already attack us and have pieces
                            # behind it which also directly attack us
                            if (board[dest_sq][1] == 1 and piece_types[0] == 3
                                and [dest_sq,1,-1] in attack_me):
                                
                                # the pawn does not block along this line
                                attack -=1
                                
                                # can we attack the pawn
                                if (our_piece in piece_types and not
                                    (pawn and dest_sq not in pawn_attacks)):
                                    attack_list.append([dest_sq,board[dest_sq][1],priority])
                                    
                                    priority -= 1
                                
                                hostile_cover = True
                                block_piece = 1
                                # skip past the rest as this pawn is not 
                                # blocking diagonal attackers    
                                continue
                            
                            # if we have already passed a blocking pawn and come
                            # to another piece that blocks the line, this is
                            # essentially the second block
                            if hostile_cover:
                                hostile_block = True
                            # if we have already passed a friendly blocking piece
                            if friendly_cover:
                                friendly_block = True
                            # END OF CODE THAT IGNORES BLOCKING PAWNS
                        
                            # however we might be attacking it
                            new_elem = [dest_sq,board[dest_sq][1],attack]
                            
                             # check if our sight of this piece is blocked
                            if friendly_block:
                                new_elem[2] = -20-block_piece
                                # check if we can attack the piece
                                if (our_piece in piece_types and not
                                    (pawn and dest_sq not in pawn_attacks)):
                                    attack_list.append(new_elem)
                                second_block = True
                            elif hostile_block:
                                new_elem[2] = -30-block_piece
                                # check if we can attack the piece
                                if (our_piece in piece_types and not
                                    (pawn and dest_sq not in pawn_attacks)):
                                    attack_list.append(new_elem)
                                second_block = True
                            else:    # the sight of the piece is not blocked
                                # check if we can attack the piece
                                if (our_piece in piece_types and not
                                    (pawn and dest_sq not in pawn_attacks)):
                                    attack_list.append([dest_sq,board[dest_sq][1],priority])
                        
                            if second_block:
                                break
                                
                            # this piece blocks along this line
                            hostile_block = True
                            second_block = True
                            block_piece = board[dest_sq][1]
                            attack = -1
                            
                else:    # the square does not contain a piece
                
                    # if there is any blocking, don't count it
                    if friendly_block or hostile_block or friendly_cover or hostile_cover:
                        continue
                    
                    # otherwise, the square is in direct view
                    
                    # check if we can attack the square
                    if (our_piece in piece_types and not
                        (pawn and dest_sq not in pawn_attacks)):
                        
                        # now check if this attack can capture en passant
                        if (our_piece == 1 and board[dest_sq][3] == True
                            and board[dest_sq+passant_sq][1] == 1):
                            attack_list.append([dest_sq,1,-1])
                        else:
                            attack_list.append([dest_sq,0,0])
                     
                    # if our piece is a king, this is a square he can be
                    # attacked from (avoid double counting on first loop)
                    if our_piece == 6 and not first_loop:
                        # defend_list.append([dest_sq,0,0])
                        attack_me.append([dest_sq,0,0])

                # finally, loop to the next move depth
                    
    # now we have finished creating our lists
    return (attack_list,attack_me,defend_me,piece_view)

def eval_piece(board,square_num,white_to_play,bonus,piece_type,piece_colour,
               original_attack_list,original_attack_me,original_defend_me):
    '''Evaluate the value of a piece'''
    
    # # extract the relevant information about the square
    # piece_type = board[square_num][1]
    # piece_colour = board[square_num][2]
    
    # # determine the attack and defend list of the piece
    # (attack_list,attack_me,defend_me) = piece_attack_defend(board,square_num,
    #                                                 piece_type,piece_colour)

    # NOT ELEGANT BUT SHOULD DO THE TRICK
    attack_list = original_attack_list[:]
    attack_me = original_attack_me[:]
    defend_me = original_defend_me[:]
    
    # OLD CODE: want to remove this as it scrambles priority lists
    # eg [[77,1,-1],[66,3,-2],[55,5,-3],[78,4,-1]]
    # the [78,4,-1] will break the order of the consecutive attack
    # # sort attack and defend lists by interaction priority
    # attack_me = sorted(attack_me, key=lambda x: x[2],reverse=True)
    # defend_me = sorted(defend_me, key=lambda x: x[2])
    
    # this sort could change order! this is important for attacks in sequence (eg rooks lined up)
    # moves are treated as sorted, but follow up attacks will NOT be correctly handled
    # with the code in its current state. THIS IS A BUG
    
    # determine the value of the piece
    if   piece_type == 1: value = 1
    elif piece_type == 2: value = 3
    elif piece_type == 3: value = 3
    elif piece_type == 4: value = 5
    elif piece_type == 5: value = 9
    elif piece_type == 6: value = 4
    
    # determine the board control of the piece ---------------------------
    attack_score = 0
    
    # assign value to the attacked squares
    if piece_colour == 1:
        target = [9,8,7,6]
    else:
        target = [2,3,4,5]
    
    centre = [3,4,5,6]
    
    for attack in attack_list:
        
        # score better for attacking squares in enemy half of board
        if attack[0]//10 in target:
            if attack[0]%10 in centre:
                attack_score += 1.5         # in their half, in the centre
            else: attack_score += 1.0       # in their half
        else:
            attack_score += 0.5             # in our half
         
        # if it attacks a piece, give another bonus
        if attack[2] in [-1,-2,-3]:
            attack_score += 1
        
        # if it is involved in a pin or discovery, even better bonus
        # SHOULD THIS BONUS BE BETTER?
        if attack[2] not in [0,-1,-2,-3]:
            attack_score += 2
    
    # convert the attack score to a bonus to increase the piece eval by
    if piece_type == 1:
        attack_bonus = bonus[0]  *  (attack_score / 2)
    elif piece_type == 2:
        attack_bonus = bonus[1]  *  (attack_score / 7)
    elif piece_type == 3:
        attack_bonus = bonus[2]  *  (attack_score / 10)
    elif piece_type == 4:
        attack_bonus = bonus[3]  *  (attack_score / 12)
    elif piece_type == 5:
        attack_bonus = bonus[4]  *  (attack_score / 21.5)
    elif piece_type == 6:
        attack_bonus = bonus[5]  *  (attack_score / 8)
        
    #print('The attack bonus is {0:.4}'.format(attack_bonus))
        
    # determine the defensive strength of the piece ----------------------
    
    net_trade = 0              # outcome of trades
    defend_piece = value       # piece on the current squares value
    king_involved = False      # is the king involved in tactics?
    
    # if the piece in question is the king, things are different
    if piece_type == 6:
        
        vunerable_from = 0
        checks = 0
        
        # add a dummy entry to the end of attack_me to avoid errors
        attack_me.append([0,0,1])
        
        # loop through the squares the king is vunerable from
        for king_attack in attack_me:
            
            # if the square is empty
            if king_attack[2] == 0:
                vunerable_from += 1
            
            # if it contains an attacking piece
            elif king_attack[2] == -1:
                checks += 1
            
        # is the king in checkmate?
        if checks != 0:
            
            # NOT ELEGANT BUT DOES THE TRICK
            copy_attack_me = original_attack_me[:]
            
            # check if it is mate using the new way
            mate = is_checkmate(board,square_num,white_to_play,bonus,
                                piece_type,piece_colour,attack_list,
                                copy_attack_me,defend_me)
            
            if mate:
                return (0,True)
            
            # FOR TESTING ---------------------------------------------------
            # double check the checkmate function
            # COMMENT if mate: above
            # old_mate = False
            # # check if its mate using the old way
            # available_moves = total_legal_moves(board,piece_colour)
            # if available_moves == []:
            #     old_mate = True

            # if mate or old_mate:
            #     if mate and old_mate:
            #         print('The old and new mate functions agree with each other!')
            #         return (0,True)
            #     else:
            #         print('FUCK the old and new checkmate functions disagree')
            #         print_board(board,False)
            #         print('The old function says',old_mate)
            #         print('The new function says',mate)
            #     # let the board evaluation know that it is mate
            #     return (0,True)
            # ----------------------------------------------------------------
        
        # has the king castled
        # at the moment hopefully no additional castle logic is needed
        # -> castle logic is handled depending on the phase
        
        # compute the vunerability of the king
        # CHECKS*3 IS MAYBE A CONSTANT THAT SHOULD BE HAND TUNED
        net_trade = -bonus[6] * (vunerable_from + (checks * 3))
        
    # else if the piece is not a king and being attacked    
    elif len(attack_me) != 0:
    
        attack_piece = 0
        
        # can you be attacked if it is your turn??
        # logic here to determine outcome if it your turn?
        
        # # are there any attackers AND is it their turn to move
        # if (attack_me[0][2] == -1 and
        #     ((white_to_play and piece_colour == 2) or
        #      (not white_to_play and piece_colour == 1))):
            
        # is it the attackers turn to move
        if ((white_to_play and piece_colour == 2) or
             (not white_to_play and piece_colour == 1)):
            
            first_loop = True
            
            # this loop determines the net trade that can occur on the
            # square, eg if a pawn attacks a knight on this square, net
            # trade after both pieces are taken is -2
            while True:
                
                i = 0
                nxt = -1
                
                # check all the possible attacks
                for attack in attack_me:
                    
                    # does this piece directly attack us
                    if attack[2] == -1:
                        
                        # if the piece value of this attack is less
                        if (nxt == -1 or attack[1] < attack_me[nxt][1]):
                            
                            nxt = i    # the next attack comes from this piece
                            
                    # # check if subsequent pieces have been uncovered (can now attack)
                    # elif attack_me[i][2] == -2 and nxt == i-1:
                    #     attack_me[i][2] = -1
                    # elif attack_me[i][2] == -3 and nxt == i-2:
                    #     attack_me[i][2] = -2
                        
                    # # check if we have reached pins etc
                    # elif attack[2] < -5:
                    #     break
                    
                    i += 1    # increment the counter
                    
                # now nxt gives the index of the next piece that will attack
                
                if first_loop:
                    # if nobody attacks us
                    if nxt == -1:
                        break
                    else:
                        first_loop = False
                else:
                    # if there are no more attackers
                    if nxt == -1:
                        net_trade += attack_piece
                        break
                    
                    # if the trade favours the defender, the attacker won't bother
                    if (attack_piece < defend_piece) or king_involved:
                        break
                    else:
                        net_trade += attack_piece
                
                # check if there are any other attackers behind this
                if (len(attack_me) < nxt+1 and attack_me[nxt+1][2] == -2):
                    attack_me[nxt+1][2] == -1
                    if (len(attack_me) < nxt+2 and attack_me[nxt+2][2] == -3):
                        attack_me[nxt+2][2] == -2
                        
                # what value does this attacker have
                if   attack_me[nxt][1] == 1:    attack_piece = 1
                elif attack_me[nxt][1] == 2:    attack_piece = 3
                elif attack_me[nxt][1] == 3:    attack_piece = 3
                elif attack_me[nxt][1] == 4:    attack_piece = 5
                elif attack_me[nxt][1] == 5:    attack_piece = 9
                elif attack_me[nxt][1] == 6:    king_involved = True
            
                # remove the attack from the list
                attack_me.pop(nxt)
                
                # # are there any more defenders
                # # if len(defend_me) != 0 and defend_me[0][2] == 1:
                # if len(defend_me) != 0:
                #     nxt = -1
                # else:    # there are no more defenders
                #     net_trade -= defend_piece
                #     break
                
                # # if the trade favours the attacker, the defender won't bother
                # if (defend_piece < attack_piece) or king_involved:
                #     break
                # else:
                #     net_trade -= defend_piece
                
                # reset and now lets check the defenders
                
                nxt = -1
                i = 0
                
                # check all the possible defenders
                for defend in defend_me:
                    
                    # does this piece directly defend us
                    if defend[2] == 1:
                        
                        # if the piece value of this defender is less
                        if (nxt == -1 or defend[1] < defend_me[nxt][1]):
                            
                            nxt = i    # the next defend comes from this piece
                            
                    # # check if subsequent pieces have been uncovered (can now defend)
                    # elif defend_me[i][2] == 2 and nxt == i-1:
                    #     defend_me[i][2] = 1
                    # elif defend_me[i][2] == 3 and nxt == i-2:
                    #     defend_me[i][2] = 2
                        
                    # # check if we have reached pins etc
                    # elif defend[2] > 5:
                    #     break
                    
                    i += 1    # increment the counter
                    
                # now nxt gives the index of the next piece that will defend
                
                # if there are no more defenders
                if nxt == -1:
                    net_trade -= defend_piece
                    break
                
                # if the trade favours the attacker, the defender won't bother
                if (defend_piece < attack_piece) or king_involved:
                    break
                else:
                    net_trade -= defend_piece
                    
                # check if there are any other defenders behind this
                if (len(defend_me) < nxt+1 and defend_me[nxt+1][2] == 2):
                    defend_me[nxt+1][2] == 1
                    if (len(defend_me) < nxt+2 and defend_me[nxt+2][2] == 3):
                        defend_me[nxt+2][2] == 2
                
                # what value does this defender have
                if   defend_me[nxt][1] == 1:    defend_piece = 1
                elif defend_me[nxt][1] == 2:    defend_piece = 3
                elif defend_me[nxt][1] == 3:    defend_piece = 3
                elif defend_me[nxt][1] == 4:    defend_piece = 5
                elif defend_me[nxt][1] == 5:    defend_piece = 9
                elif defend_me[nxt][1] == 6:    king_involved = True
                
                # remove the defend from the list
                defend_me.pop(nxt)
                
                # # are there any more attackers
                # if len(attack_me) != 0 and attack_me[0][2] == -1:
                #     nxt = 0
                # else:    # there are no more attackers
                #     net_trade += attack_piece
                #     break
                
                # # if the trade favours the defender, the attacker won't bother
                # if (attack_piece < defend_piece) or king_involved:
                #     break
                # else:
                #     net_trade += attack_piece
                    
    else:    # the piece is not attacked
    
        net_trade = 0
        
        # logic here to compute value of discoveries/pins etc?
        # logic here to reward pieces that have more defenders than attackers?
        
    #print('The net trade is ',net_trade)
    
    # the net trade has now been calculated
    # the attack weight has now been calculated
    # time to determine the overall value of the piece
    # need to adjust sign too, since -ve means against player atm
    
    if piece_colour == 1:
        sign = 1
    else:
        sign = -1
        
    final_value = (value + attack_bonus + net_trade) * sign
    
    #print('The final value of the ',name,' is {0:.4}'.format(final_value))
    
    # end of the eval_piece function
    return (final_value,False)
            

def eval_board_2(board,white_to_play):
    '''This function evaluates the board to decide who is winning''' 
    
    # Start of the eval_board function
        
    # set the starting evaluation as even
    evaluation = 0.0
    
    # determine the phase of the game, piece bonuses and resultant evaluation
    phase,bonus,value = determine_phase(board,white_to_play)
    
    # update the evaluation based on the phase of play
    evaluation += value
    
    # loop through every square of the board
    for i in range(8):
        for j in range(8):
            index = (i+2)*10  +  (j+1)
            
            # if the square contains a piece, find its evaluation
            if board[index][1] != 0:
    
                # extract the relevant information about the square
                piece_type = board[index][1]
                piece_colour = board[index][2]
    
                # determine the attack and defend list of the piece
                (attack_list,attack_me,
                 defend_me,piece_view) = piece_attack_defend(board,index,
                                                       piece_type,piece_colour)
                
                ## determine the value of the piece
                #if   piece_type == 1: value = 'pawn'
                #elif piece_type == 2: value = 'knight'
                #elif piece_type == 3: value = 'bishop'
                #elif piece_type == 4: value = 'rook'
                #elif piece_type == 5: value = 'queen'
                #elif piece_type == 6: value = 'king'
                
                # print('The',value,'on square',index,'has:')
                # print('attack_list',attack_list)
                # print('attack_me',attack_me)
                # print('defend_me',defend_me)
                
                (value,mate) = eval_piece(board,index,white_to_play,bonus,
                                          piece_type,piece_colour,attack_list,
                                          attack_me,defend_me)
                
                # if there is a checkmate in the position WHAT ABOUT DRAWS FROM KING CAN'T MOVE
                if mate:
                    if white_to_play:
                        return -100.1
                    else:
                        return 100.1
                
                # update the evaluation with the piece value
                evaluation += value
            
    return evaluation

def determine_phase(board,white_to_play):
    '''This function determines the phase of play on a board, and outputs
    the phase and the relative piece bonuses'''
    
    def tempo_check(board,white_to_play):
        '''This function checks who is ahead in development'''
        
        if white_to_play:
            tempo = 0
        else:
            tempo = -1
            
        # white pieces
        # if board[21][1] == 4: tempo -= 1
        if board[22][1] == 2: tempo -= 1
        if board[23][1] == 3: tempo -= 1
        if board[24][1] == 6: tempo -= 1
        # skip the queen to discourage her use early on
        if board[26][1] == 3: tempo -= 1
        if board[27][1] == 2: tempo -= 1
        # if board[28][1] == 4: tempo -= 1
        
        # white pawns
        #if board[31][1] == 1: tempo -= 1
        #if board[32][1] == 1: tempo -= 1
        # if board[33][1] == 1: tempo -= 1
        if board[34][1] == 1: tempo -= 1
        if board[35][1] == 1: tempo -= 1
        if board[36][1] == 1: tempo -= 1
        #if board[37][1] == 1: tempo -= 1
        #if board[38][1] == 1: tempo -= 1
        
        # black pieces
        # if board[91][1] == 4: tempo += 1
        if board[92][1] == 2: tempo += 1
        if board[93][1] == 3: tempo += 1
        if board[94][1] == 6: tempo += 1
        # skip the queen to discourage her use early on
        if board[96][1] == 3: tempo += 1
        if board[97][1] == 2: tempo += 1
        # if board[98][1] == 4: tempo += 1
        
        # black pawns
        #if board[81][1] == 1: tempo += 1
        #if board[82][1] == 1: tempo += 1
        # if board[83][1] == 1: tempo += 1
        if board[84][1] == 1: tempo += 1
        if board[85][1] == 1: tempo += 1
        if board[86][1] == 1: tempo += 1
        #if board[87][1] == 1: tempo += 1
        #if board[88][1] == 1: tempo += 1
        
        return tempo
    
    def rate_pawn_structure(board):
        '''This function gives points to each side based on their pawn structure,
        extra points for advanced pawns, protected pawns and passed pawns'''
        
        white = []
        black = []
        cols = [ [[],[]] for i in range(8)]
        
        # go throught the whole board and get the square of every pawn
        for i in range(8):
            for j in range(8):
                # this goes A8->A1, then B8->B1 etc
                index = (i+2)*10  + (j+1)
                if board[index][1] == 1:
                    if board[index][2] == 1:
                        white.append(index)
                        cols[j][0].append(index)
                    else:
                        black.append(index)
                        cols[j][1].append(index)
                        
        # now we have the squares of all the white and black pawns
        
        # go through the white pawns and score them
        white_score = 0
        for pawn_sq in white:
            
            # what row is this pawn (zero indexing so from 0-7)
            row = (pawn_sq // 10) - 2
            # give one point per row it has advanced
            white_score += row-1
            
            # what column is this pawn (zero indexing so from 0-7)
            column = (pawn_sq % 10) - 1
            # is this pawn protected by another pawn
            # give two points for a protected pawn
            if column - 1 >= 0 and pawn_sq - 11 in cols[column-1][0]:
                white_score += 2
            if column + 1 <= 7 and pawn_sq - 9 in cols[column+1][0]:
                white_score += 2
                
            # is this pawn a passed pawn
            passed = True
            for n in range(-1,2):
                if column+n >= 0 and column+n <= 7:
                    for m in cols[column+n][1]:
                        if m > (row+3)*10:
                            passed = False
                            break
                if passed == False:
                    break
            # give 5 points for a passed pawn and boost by its row
            # so passed pawn on second rank is 5pts (+0.5 if divided by 10)
            # then passed pawn on seventh rank is 20pts (+2.0 if divided by 10)
            # this multiplier incentivises pushing passed pawns
            if passed:
                white_score += 5 + (row-1)*3
        
        # now go through the black pawns
        black_score = 0
        for pawn_sq in black:
            
            # what row is this pawn (zero indexing so from 0-7)
            row = (pawn_sq // 10) - 2
            # give one point per row
            black_score += 6-row
            
            # what column is this pawn (zero indexing so from 0-7)
            column = (pawn_sq % 10) - 1
            # is this pawn protected by another pawn
            # give two points for a protected pawn
            if column - 1 >= 0 and pawn_sq + 9 in cols[column-1][1]:
                black_score += 2
            if column + 1 <= 7 and pawn_sq + 11 in cols[column+1][1]:
                black_score += 2
                
            # is this pawn a passed pawn
            passed = True
            for n in range(-1,2):
                if column+n >= 0 and column+n <= 7:
                    for m in cols[column+n][0]:
                        if m < (row+2)*10:
                            passed = False
                            break
                if passed == False:
                    break
            # give 5 points for a passed pawn and boost by its row
            # so passed pawn on second rank is 5pts (+0.5 if divided by 10)
            # then passed pawn on seventh rank is 20pts (+2.0 if divided by 10)
            # this multiplier incentivises pushing passed pawns
            if passed:
                black_score += 5 + (6-row)*3
                
        # what is the final score
        final_score = white_score - black_score
        
        # scale this score by dividing by 20 (ie passed pawn = +0.5, final rank = +0.3)
        final_value = final_score / 10.0
        
        # end of rate_pawn_structure function
        return final_value
    
    phase = 1
    
    # if both players have castled, or given up rights, phase 2
    if (board[5][3] == 3 or
        not (board[0][3] or board[1][3] or board[2][3] or board[3][3])):
        phase = 2
        
        # if enough pieces have left the board, phase 3
        #if np.sum(np.array(board)[:,1]) < 32:
            #phase = 3
    
    # # extract what phase of play we are in
    # phase = board[100][3]
    
    # what phase of play is it
    if phase == 1:
        # bonus: pwn  knight bish  rook queen  king  king(vunerability)
        bonus = [0.25, 0.40, 0.40, 0.10, 0.25, 0.25, 0.05]
    elif phase == 2:
        bonus = [0.25, 0.50, 0.50, 0.50, 0.50, 0.25, 0.10]
    elif phase == 3:
        bonus = [0.50, 0.50, 0.50, 0.75, 0.75, 1.00, 0.10]
        
    # apply some checks based on the phase of play
    
    evaluation = 0.0
    
    # Castle Rights
    
    if phase == 1 or phase == 2:
        # if white has castled but black hasn't
        if board[5][3] == 1:
            evaluation += 0.3    # reward white
            
            # if black is unable to castle, punish black
            if board[2][3] == False:
                evaluation += 0.2
            if board[3][3] == False:
                evaluation += 0.2
        
        # if black has castled but white hasn't
        elif board[5][3] == 2:
            evaluation -= 0.3    # reward black
            
            # if white is unable to castle, punish white
            if board[0][3] == False:
                evaluation -= 0.2
            if board[1][3] == False:
                evaluation -= 0.2
                
        elif board[5][3] == 0:
            
            # if black is unable to castle, punish black
            if board[2][3] == False:
                evaluation += 0.2
            if board[3][3] == False:
                evaluation += 0.2
                
            # if white is unable to castle, punish white
            if board[0][3] == False:
                evaluation -= 0.2
            if board[1][3] == False:
                evaluation -= 0.2
                
        # next, penalise the player with more pieces on their starting squares
        tempo = tempo_check(board,white_to_play)
        
        # adjust the evaluation based off the tempo weighting
        evaluation += tempo * 0.3
        
        # # REMOVE THIS DURING TESTING OF BEST_MOVES_PLUS
        # # also adjust based on who is playing next, assuming they will improve
        # if white_to_play:
        #     evaluation += 0.1
        # else:
        #     evaluation -= 0.1
        
        # test! rate_pawn_structure slows things down by about 5% (1sec / 20 total)
        # pawn_value = rate_pawn_structure(board)
    
    # add logic here for middlegame, pawn structure, passed pawns etc 
    elif phase == 2:
        pawn_value = rate_pawn_structure(board)
        evaluation += pawn_value
    elif phase == 3:
        pawn_value = rate_pawn_structure(board)
        evaluation += pawn_value
        
    return phase,bonus,evaluation

#-----------------------------------------------------------------------------#
def total_legal_moves(board,player_colour=1):
    '''This function generates a list of every legal move that can be made
    by a single player'''
                      
    total_legal_moves = []
    
    for i in range(8):
        for j in range(8):
            
            index = (2+i)*10 + (j+1)
            
            if board[index][2] == player_colour:
                move_list = legal_moves(board,index)
                if move_list != []:
                    total_legal_moves.append([index,[]])
                    p = len(total_legal_moves)-1
                    for m in range(len(move_list)):
                        total_legal_moves[p][1].append(move_list[m])
                    
    return total_legal_moves
    
#-----------------------------------------------------------------------------#

def total_legal_moves_adjusted(board,player_colour=1):
    '''This function generates a list of every legal move that can be made
    by a single player'''
                      
    total_legal_moves = []
    
    for i in range(8):
        for j in range(8):
            
            index = (2+i)*10 + (j+1)
            
            if board[index][2] == player_colour:
                move_list = legal_moves(board,index)
                if move_list != []:
                    for m in range(len(move_list)):
                        new_elem = [index,move_list[m][0],move_list[m][1]]
                        total_legal_moves.append(new_elem)
                    
                    # total_legal_moves.append([index,[]])
                    # p = len(total_legal_moves)-1
                    # for m in range(len(move_list)):
                    #     total_legal_moves[p][1].append(move_list[m])
                    
    return total_legal_moves
    
#-----------------------------------------------------------------------------#
        
def ind_to_letter(board_index):
    '''This method takes a board index [start_sq,dest_sq] and converts it
    to standard letter format eg e2e4'''
    
    start_sq = board_index[0]
    dest_sq = board_index[1]
    
    if start_sq == 0 and dest_sq == 0:
        return 'none'

    # figure out what the move string is
    new_str = []
    board_str = str(start_sq)
    
    try:
    
        if   board_str[1] == '1':  new_str = 'h'   
        elif board_str[1] == '2':  new_str = 'g'
        elif board_str[1] == '3':  new_str = 'f'
        elif board_str[1] == '4':  new_str = 'e'
        elif board_str[1] == '5':  new_str = 'd'
        elif board_str[1] == '6':  new_str = 'c'
        elif board_str[1] == '7':  new_str = 'b'
        elif board_str[1] == '8':  new_str = 'a'
        if   board_str[0] == '2':  new_str += '1'
        elif board_str[0] == '3':  new_str += '2'
        elif board_str[0] == '4':  new_str += '3'
        elif board_str[0] == '5':  new_str += '4'
        elif board_str[0] == '6':  new_str += '5'
        elif board_str[0] == '7':  new_str += '6'
        elif board_str[0] == '8':  new_str += '7'
        elif board_str[0] == '9':  new_str += '8'
        
    except:
        
        print('ERROR in ind_to_letter, failed on input ',board_index)
        quit()
    
    new_str2 = []
    board_str = str(dest_sq)
    if   board_str[1] == '1':  new_str2 = 'h'   
    elif board_str[1] == '2':  new_str2 = 'g'
    elif board_str[1] == '3':  new_str2 = 'f'
    elif board_str[1] == '4':  new_str2 = 'e'
    elif board_str[1] == '5':  new_str2 = 'd'
    elif board_str[1] == '6':  new_str2 = 'c'
    elif board_str[1] == '7':  new_str2 = 'b'
    elif board_str[1] == '8':  new_str2 = 'a'
    if   board_str[0] == '2':  new_str2 += '1'
    elif board_str[0] == '3':  new_str2 += '2'
    elif board_str[0] == '4':  new_str2 += '3'
    elif board_str[0] == '5':  new_str2 += '4'
    elif board_str[0] == '6':  new_str2 += '5'
    elif board_str[0] == '7':  new_str2 += '6'
    elif board_str[0] == '8':  new_str2 += '7'
    elif board_str[0] == '9':  new_str2 += '8'
    
    # so the move string is
    move_str = new_str + new_str2
    
    return move_str

def checkmate(board,white_to_play):
    '''This function checks if the person who plays next is in checkmate'''
    
    if white_to_play:
        list_of_moves = total_legal_moves(board,1)
    else:
        list_of_moves = total_legal_moves(board,2)
     
    # if there are no legal moves it is checkmate
    if list_of_moves == []:
        return True
    else:
        return False
    
def total_legal_moves_plus(board,white_to_play):
    '''This function gets all the legal moves for one player using the
    piece_attack_defend function to rank the quality of those moves'''
    
    # this function should use piece_attack_defend to get the squares every piece
    # can move to, and at the same time get the evalutions of every piece and
    # store them in a pseudo board matrix with the correct lookup index, in fact
    # this pseudo board should also include the piece_attack_defend outputs for each
    # piece, or more specifically the VIEW from each piece square
    # this function should also produce seperate lists for checks, captures,
    # moves etc ie it sorts moves based on priority
    # it is important to return the move modifier for each move too
    # don't forget en passant and castling
    # currently total_legal_moves sets promote+check as a check, perhaps better
    # to make a new move modifier? maybe multiple promote moves, you have
    # promote to queen, promote to knight, promote to queen with check and 
    # promote to knight with check
    
    # okay so what do I need
    # the evaluation of every piece and the corresponding square
    # the view of every piece, which is all pieces that can interact with it
    # the legal moves of every piece
    
    # for legal moves, for all except pawns it is the squares they attack and
    # any unblocked captures
    
    # the only time a non-king move can lead to check is if the piece is pinned
    # to the king
    
    # the king can only go to squares NOT controlled by enemy pieces
    
    # so far: I have done pawns (including en passant) and I have done regular
    # pieces, all that remains is to compute legal moves for the king and crucially
    # if the king is in check or certain squares are enemy controlled
    
    # NB since 6 and 7 now mean 'promote to knight' and 'promote to queen' I should
    # update the make_move function - should I include rook and bishop...? not for computer
    # but probably use 8/9 for that and give human the option
    
    if white_to_play:
        player_colour = 1
        pawn_move = 10
        pawn_start = 3
        pawn_promote = 9
    else:
        player_colour = 2
        pawn_move = -10
        pawn_start = 8
        pawn_promote = 2
    
    # data array has field: piece_eval,attack_list,attack_me,defend_me,piece_view
    data_array = [ [0.0,[],[],[],[]] for p in range(65) ]
    
    our_indexes = []
    their_indexes = []
    
    #checks = []    # checks and promotions NOT CURRENTLY IMPLEMENTED
    captures = []  # piece captures and castles
    norm_moves = []     # regular piece moves
    
    # determine the phase of the game, piece bonuses and resultant evaluation
    phase,bonus,phase_value = determine_phase(board,white_to_play)
    
    data_array[64][0] = phase_value
    
    # start recording an evaluation for the position
    evaluation = phase_value
    
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
                defend_me,piece_view) = piece_attack_defend(board,index,
                                                           piece_type,
                                                           piece_colour)
                
                # evaluate pieces from NOT white_to_play because this identifies
                # pieces that we have which are hanging
                (value,mate) = eval_piece(board,index,not white_to_play,bonus,
                                          piece_type,piece_colour,attack_list,
                                          attack_me,defend_me)
                
                # don't forget, attack_me/attack_list etc have elements removed
                # in eval_piece. This doesn't seem to be a problem currently
 
                # if there is a checkmate in the position
                # WHAT ABOUT DRAWS FROM KING CAN'T MOVE
                # note that eval_piece uses total_legal_moves for checkmate
                # detection, which is very slow
                if mate:
                    outcome = 1
                    if white_to_play:
                        return [],[],-100.1,outcome
                    else:
                        return [],[],100.1,outcome
                
                # detect the king, to deal with first
                if piece_type == 6:
                    if piece_colour == player_colour:
                        king_index = index
                    
                # is it our piece
                elif piece_colour == player_colour:
                    our_indexes.append(index)
                    
                else:    # it is one of their pieces
                    their_indexes.append(index)
                    
                # save the information in the data array
                data_array[(i*8)+j][0] = value
                data_array[(i*8)+j][1] = attack_list
                data_array[(i*8)+j][2] = attack_me
                data_array[(i*8)+j][3] = defend_me
                data_array[(i*8)+j][4] = piece_view
                
                # update the evaluation with the piece value
                evaluation += value
                    
    # now we have finished filling the data array
    
    # first, we should deal with the king
    # find if the king is in check, since it limits legal moves
    # use king_index
    # don't forget to add castling!
    
    # check if our king is in check
    num_checks = 0
    pinned_piece = []
    pinned_moves = []
    block_check = []
    check = False
    pin = False
    
    maybe_pin = False
    
    try:
        # find the corresponding data array index
        k_ind = (king_index - 21) - 2*((king_index//10) - 2)
    except:
        print_board(board,False)
        throw_error()
    
    # loop through any attacks to the king
    for attack in data_array[k_ind][2]:
        
        # if a piece is checking us
        if attack[2] == -1:
            num_checks += 1
            check = True
          
        # if a piece is pinned to us (nb -21//10 = -3)
        elif attack[2] // 10 == -3:
            maybe_pin = True
            
        else:    # it is an attack blocked by a friendly piece
            continue
            
        attack_line = attack[0] - king_index
        
        direction = 0
            
        for q in range(1,8):
            
            if attack_line == q:
                direction = 1
                break
            elif attack_line == -q:
                direction = -1
                break
            elif attack_line == -9*q:
                direction = -9
                break
            elif attack_line == -10*q:
                direction = -10
                break
            elif attack_line == -11*q:
                direction = -11
                break
            elif attack_line == 9*q:
                direction = 9
                break
            elif attack_line == 10*q:
                direction = 10
                break
            elif attack_line == 11*q:
                direction = 11
                break
            
        # if direction is not set, it must be knight check 
        if direction == 0:
            # knight checks cannot be blocked, only the knight captured
            block_check.append(attack[0])
            continue

        move_set = []
        num_pinned = 0
        
        # now we know what line the piece is pinned along
        for l in range(1,8):
            
            # if the square is out of bounds, break
            if board[king_index+direction*l][0] == False:
                break
            
            # if the square is empty along the pin/check line
            elif board[king_index+direction*l][1] == 0:
                
                # add this square to the set
                move_set.append(king_index+direction*l)
            
            # if we reach the piece that is pinning/checking us
            elif (board[king_index+direction*l][1] == attack[1] and
                  board[king_index+direction*l][2] != player_colour):
                # save the square from which the pin/check comes, then break
                move_set.append(king_index+direction*l)
                break
            
            # if we reach the piece that is pinned
            elif (attack[2] // 10 == -3 and
                  board[king_index+direction*l][2] == player_colour):
                pinned_piece.append(king_index+direction*l)
                num_pinned += 1
            
        # now, we have all the squares along the line
        
        # if it is a checking line
        if attack[2] == -1:
            block_check += move_set
            
        # if it is a pin
        elif attack[2] // 10 == -3:
            # if we do indeed have a pinned piece
            if num_pinned == 1:
                pin = True
                pinned_moves.append(move_set)
            elif num_pinned == 0:
                # this should never happen
                print('We were told there was a pin but didnt find a pinned piece')
            # if we found multiple pinned pieces, its not really a pin!
            else:
                # remove everything we added in error
                for oops in range(num_pinned):
                    pinned_piece.pop()
            # reset
            maybe_pin = False
            
        else:
            print('Shouldnt be here in total_legal_moves_plus')
            
    # next, we are going to see what legal moves the king can make
    
    # # firstly, remove the king from his square (so it can't block any checks)
    board[king_index][1:3] = [0,0]
            
    # now, loop through any squares the king can move to to determine if they are safe
    for squares in data_array[k_ind][1]:
        
        # is this square controlled by an opposing piece
        (attack_list,attack_me,
         defend_me,piece_view) = piece_attack_defend(board,squares[0],
                                                           6,player_colour)
                                                     
        unsafe = False
        
        # loop through any attacks on this square
        for attack in attack_me:
            
            # if the square is attacked, it is not safe to move to
            if attack[2] == -1:
                unsafe = True
                break
        
        # if the square is safe, the king can move there
        if not unsafe:
            if squares[2] == 0:
                norm_moves.append([king_index,squares[0],1])
            elif squares[2] == -1:
                captures.append([king_index,squares[0],2])
            else:
                print('argghh shouldnt be here either in total_legal_moves_plus')
                
    # # we have finished checking if the king has any moves, return him to the board
    board[king_index][1:3] = [6,player_colour]
    
    # determine if there are any castling moves that can be made
    castles = can_castle(board,player_colour)
    
    # if there are, add them to the move list (in captures for higher priority)
    for moves in castles:        
        captures.append([king_index,moves[0],moves[1]])
            
    # print('Block check is ',block_check)
    # print('pinned moves is ',pinned_moves)
    # print('Check is',check,'and pin is',pin)
    # print('pinned piece is ',pinned_piece)
   

            
    # if yes, determine the squares inbetween  (ie the lines) along which the 
    # king is checked, other piece moves may only end on the line (if there is
    # only one) or capture the checking piece (if there is only one)
    # if its a double check, the king must move
    
    # assess every adjacent square to the king to see if it is enemy controlled
    
    # find if any pieces are pinned to the king, in which case they can not
    # move
    
    # get a list of squares the king can move to
    # if in check, get a list of squares other pieces can move to/capture on
    # ie a selection of legal moves eg [33,34,1] or [33,43,2]
    
    # don't forget to add in possible castles
    
    # next, go through the rest of our the pieces
    for ind in our_indexes:
        
        # if it is a double check, other pieces have no legal moves
        if num_checks > 1:
            continue
        
        # find the corresponding data array index
        d_ind = (ind - 21) - 2*((ind//10) - 2)
        
        # if the piece is a pawn, then it has different legal moves
        if board[ind][1] == 1:
            
            i_am_pinned = False
            
            # if there is a pin on the board
            if pin:
                # loop through to see if this piece is pinned
                for p in range(len(pinned_piece)):
                    if ind == pinned_piece[p]:
                        i_am_pinned = True
                        break
            
            for attack in data_array[d_ind][1]:
                
                # if we are pinned
                if i_am_pinned:
                    if attack[0] not in pinned_moves[p]:
                        continue
                
                # if our king is in check, we can only block or capture
                if check:
                    if attack[0] not in block_check:
                        continue
                    
                # if a piece is attacked, we can move there
                if attack[2] == -1:
                    # if the destination square is empty, it is en passant
                    if board[attack[0]][1] == 0:
                        move_mod = 3
                        captures.append([ind,attack[0],move_mod])
                    # do we promote with this capture
                    elif attack[0]//10 == pawn_promote:
                        # 6: promote to knight
                        # 7: promote to queen
                        # 8: promote to bishop
                        # 9: promote to rook
                        captures.append([ind,attack[0],6])
                        captures.append([ind,attack[0],7])
                        captures.append([ind,attack[0],8])
                        captures.append([ind,attack[0],9])
                    else:    # it is a normal capture
                        move_mod = 2
                        captures.append([ind,attack[0],move_mod])
                    
            # now, if the square ahead of us is empty and we can move there
            if (board[ind+pawn_move][1] == 0 and
                (not check or ind+pawn_move in block_check) and not
                (i_am_pinned and ind+pawn_move not in pinned_moves[p])):
                # is this move a promotion?
                if (ind+pawn_move)//10 == pawn_promote:
                    # 6: promote to knight
                    # 7: promote to queen
                    # 8: promote to bishop
                    # 9: promote to rook
                    captures.append([ind,ind+pawn_move,6])
                    captures.append([ind,ind+pawn_move,7])
                    captures.append([ind,ind+pawn_move,8])
                    captures.append([ind,ind+pawn_move,9])
                else:    # it is not a promotion
                    move_mod = 1
                    norm_moves.append([ind,ind+pawn_move,move_mod])
            
            # if we haven't moved yet and the criteria is met for moving
            if (ind//10 == pawn_start and
                board[ind+pawn_move][1] == 0 and
                board[ind+2*pawn_move][1] == 0 and
                (not check or ind+2*pawn_move in block_check) and not
                (i_am_pinned and ind+2*pawn_move not in pinned_moves[p])):
                # we can do a double move
                move_mod = 1
                norm_moves.append([ind,ind+2*pawn_move,move_mod])                           
        
        else: # it is not a pawn or a king
        
            i_am_pinned = False
            
            # if there is a pin on the board
            if pin:
                # loop through to see if this piece is pinned
                for p in range(len(pinned_piece)):
                    if ind == pinned_piece[p]:
                        i_am_pinned = True
                        break
            
            # for each attack in the attack list
            for attack in data_array[d_ind][1]:
                
                # if our king is in check, we can only block or capture
                if check:
                    if attack[0] not in block_check:
                        continue
                    
                # if we are pinned
                if i_am_pinned:
                    if attack[0] not in pinned_moves[p]:
                        continue
                
                # if a square is attacked (ie we can move there)
                if attack[2] == 0:
                    move_mod = 1
                    norm_moves.append([ind,attack[0],move_mod])
                 
                # if a piece is attacked (ie we can capture)
                elif attack[2] == -1:
                    move_mod = 2
                    captures.append([ind,attack[0],move_mod])
     
    # captures now represents the total legal moves
    captures += norm_moves
    
    # we can now determine whether it is checkmate or a draw via no legal moves
    if len(captures) == 0:
        
        # if we are in check, it must be checkmate
        if check:
            outcome = 1    # checkmate
        
        # if we aren't in check, it must be a draw
        else:
            outcome = 2    # draw
            
    else:    # the game is not over
    
        outcome = 0    # game continues
                            
    return captures,data_array,evaluation,outcome


def board_outcome():
    '''This function determines if the board is drawn, won by either side, or
    still undetermined'''
    
    pass

def is_checkmate(board,square_num,white_to_play,bonus,piece_type,player_colour,
               attack_list,attack_me,defend_me):
    '''This function determines whether the king is in checkmate'''
    
    if piece_type != 6:
        print('is_checkmate has been asked about a piece that isnt a king')
        return False
    
    # now we have finished filling the data array
    
    # # save the information in the data array
    # data_array[(i*8)+j][0] = value
    # data_array[(i*8)+j][1] = attack_list
    # data_array[(i*8)+j][2] = attack_me
    # data_array[(i*8)+j][3] = defend_me
    # data_array[(i*8)+j][4] = piece_view
    
    # first, we should deal with the king
    # find if the king is in check, since it limits legal moves
    # use king_index
    # don't forget to add castling!
    
    king_index = square_num
    
    # check if our king is in check
    num_checks = 0
    pinned_piece = []
    pinned_moves = []
    block_check = []
    check = False
    pin = False
    
    maybe_pin = False
    
    # loop through any attacks to the king
    for attack in attack_me:
        
        # if a piece is checking us
        if attack[2] == -1:
            num_checks += 1
            check = True
          
        # if a piece is pinned to us (nb -21//10 = -3)
        elif attack[2] // 10 == -3:
            maybe_pin = True
            
        else:    # it is an attack blocked by a friendly piece
            continue
            
        attack_line = attack[0] - king_index
        
        direction = 0
            
        for q in range(1,8):
            
            if attack_line == q:
                direction = 1
                break
            elif attack_line == -q:
                direction = -1
                break
            elif attack_line == -9*q:
                direction = -9
                break
            elif attack_line == -10*q:
                direction = -10
                break
            elif attack_line == -11*q:
                direction = -11
                break
            elif attack_line == 9*q:
                direction = 9
                break
            elif attack_line == 10*q:
                direction = 10
                break
            elif attack_line == 11*q:
                direction = 11
                break
            
        # if direction is not set, it must be knight check 
        if direction == 0:
            # knight checks cannot be blocked, only the knight captured
            block_check.append(attack[0])
            continue

        move_set = []
        num_pinned = 0
        
        # now we know what line the piece is pinned along
        for l in range(1,8):
            
            # if the square is out of bounds, break
            if board[king_index+direction*l][0] == False:
                break
            
            # if the square is empty along the pin/check line
            elif board[king_index+direction*l][1] == 0:
                
                # add this square to the set
                move_set.append(king_index+direction*l)
            
            # if we reach the piece that is pinning/checking us
            elif (board[king_index+direction*l][1] == attack[1] and
                  board[king_index+direction*l][2] != player_colour):
                # save the square from which the pin/check comes, then break
                move_set.append(king_index+direction*l)
                break
            
            # if we reach the piece that is pinned
            elif (attack[2] // 10 == -3 and
                  board[king_index+direction*l][2] == player_colour):
                pinned_piece.append(king_index+direction*l)
                num_pinned += 1
            
        # now, we have all the squares along the line
        
        # if it is a checking line
        if attack[2] == -1:
            block_check += move_set
            
        # if it is a pin
        elif attack[2] // 10 == -3:
            # if we do indeed have a pinned piece
            if num_pinned == 1:
                pin = True
                pinned_moves.append(move_set)
            elif num_pinned == 0:
                # this should never happen
                print('We were told there was a pin but didnt find a pinned piece')
                exit()    # throw an error
            # if we found multiple pinned pieces, its not really a pin!
            else:
                # remove everything we added in error
                for oops in range(num_pinned):
                    pinned_piece.pop()
            # reset
            maybe_pin = False
            
        else:
            print('Shouldnt be here in total_legal_moves_plus')
            
    # so are we in check
    if not check:
        # if not, it cannot be checkmate
        return False
    
    # print('attack list is',attack_list)
    # print('block check is',block_check)
    # print('pinned moves is',pinned_moves)
    
    # we will now loop through all the squares adjacent to the king
    
    # firstly, remove the king from his square (so it can't block any checks)
    board[king_index][1:3] = [0,0]
            
    # now, loop through any squares the king can move to to determine if they are safe
    for squares in attack_list:
        
        # is this square controlled by an opposing piece
        (new_attack_list,new_attack_me,
         new_defend_me,piece_view) = piece_attack_defend(board,squares[0],
                                                           6,player_colour)
                                                     
        unsafe = False
        
        # loop through any attacks on this square
        for attack in new_attack_me:
            
            # if the square is attacked, it is not safe to move to
            if attack[2] == -1:
                unsafe = True
                break
        
        # if the square is safe, the king can move there
        if not unsafe:
            # so it is not checkmate
            # print('Not checkmate, the king can move to square',squares[0])
            
            # return the king to his square before leaving
            board[king_index][1:3] = [6,player_colour]
            return False
        
    # we have finished checking if the king has any moves, return him to the board
    board[king_index][1:3] = [6,player_colour]
        
    # if it is a double check, other pieces have no legal moves
    if num_checks > 1:
        # hence, it must be checkmate - since we must have no legal king moves
        return True
                
    # if there is only one check, loop through the check line squares to see
    # if we have pieces that can block the check/attack along that line
    
    if player_colour == 1:
        pawn_move = 10
        pawn_start = 3
    else:
        pawn_move = -10
        pawn_start = 8
    
    for square in block_check:
        
        # determine who controls this square
        (new_attack_list,new_attack_me,
         new_defend_me,piece_view) = piece_attack_defend(board,square,
                                                           6,player_colour)
                                                         
        # first, check if any pawns can move to the square
        # print(piece_view)
        
        # is this square empty and therefore a pawn could move to it
        if board[square][1] == 0:
        
            # what are the squares a pawn would have to move from
            pawn_squares = [square + pawn_move*-1]
            
            # could a pawn do a double move to get here?
            if (square + pawn_move*-2) // 10 == pawn_start:
                if board[square+pawn_move*-1][1] == 0:
                    pawn_squares.append(square + pawn_move*-2)
                    
            for view in piece_view:
                if view in pawn_squares:
                    if board[view][1] == 1 and board[view][2] == player_colour:
                        # print('not checkmate since the pawn on',view,'can move to square',square)
                        return False
        
        # check each of our pieces that can travel/defend this check blocking square        
        for defend in new_defend_me:
                
            i_am_pinned = False
            
            # check the piece in question is not the king! It can't block itself
            if defend[1] == 6:
                continue
            
            # now check if it is a pawn
            if defend[1] == 1:
                # check if the square contains an opposing piece
                if board[square][2] in [0,player_colour]:
                    # if not, a pawn can't move to an attacked square
                    continue
            
            # if there is a pin on the board
            if pin:
                # loop through to see if this piece is pinned
                for p in range(len(pinned_piece)):
                    if defend[0] == pinned_piece[p]:
                        i_am_pinned = True
                        break
                
            # if we are pinned
            if i_am_pinned:
                if square not in pinned_moves[p]:
                    continue
                
            # hence if we aren't pinned, the move is legal if it is a legit defend
            if defend[2] == 1:
                # print('Not checkmate, the piece on',defend[0],'can move to square',square)
                return False
     
    # if we get here, we must indeed have no legal moves, it is checkmate
    return True
    
def legal_king_moves():
    '''This function determines if the king has any legal moves, to aid in
    finding both checkmates and draws'''
                        
    pass                  

    