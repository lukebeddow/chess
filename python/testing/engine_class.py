# This class is a chess engine! It calculates moves to play

import board_light as bl
# import board_numpy as bl

import numpy as np
import copy

#import global_variables

# for testing ----------------------
import sys
sys.path.insert(0, "C:\\Users\\lukeb\\Documents\\Chess\\Visual studio\\Chess Board Functions\\Release")
import board_func as bf
import cpp_py_conversions_vs as cpc
# ----------------------------------

import time

class chess_engine:
    
    def __init__(self,move_order=[],opening_book=[],delete_old=[],width_decay=False):
        '''Initialise the engine'''
        
        # this is the tolerance for a positive ID comparison of two dp floats
        self.tolerance = 1e-15
        self.mode = "testing"
        
        # create a new board and evaluate it
        if move_order == []:
            new_board = bl.create_board()
            board_eval = 0.0
        else:
            if type(move_order[0]) == str:
                move_order = self.letters_to_moves(move_order)
                
            if len(move_order) % 2 == 0:
                white_to_play = True
            else:
                white_to_play = False
            
            new_board = bl.create_board(move_order)
            board_eval = bl.eval_board_2(new_board,white_to_play)
            
        # find the hash ID of the board
        hash_id = hash(str(new_board[:100]))
        
        # save the board and evaluation in data structures
        self.board_array_list = [[new_board]]
        self.eval_array_list = [[[board_eval]]]
        self.id_array_list = [[[hash_id]]]
        
        self.parent_id_list = [[[ ]]]
        
        self.all_board_moves = [[[ ]]]
        
        self.active_node_array = [[[ ]]]
        
        self.active_node_evals = [[[]]]
        
        self.num_nodes_array = [[[1]]]
        
        self.sorted_hash_array = [[0,hash_id]]
        self.sorted_hash_keys = [[0]]
        
        self.offset = 0     # move offset for handling deleted lists
        
        self.use_width_decay = width_decay
        
        self.reactivate_list = []    # empty list for nodes that should be reactivated
        
        # data for handling the 'check remaining branches method'
        self.old_evaluations = []
        self.old_eval_id = [-1,0]    # an impossible initial value
        
        # for timings - this needs to be calibrated! Updated at end of depth search
        #               so only matters for very first search
        self.avg_board_ms = 15
        
        # for threading
        self.threading = False
        
        # set the parameters for allowance decay
        self.decay_min = 0.2         # decay tends to this value (default = 0.2)
        self.decay_const = 0.5       # rate of decay (default = 0.5)
        
        # decay_const     value after k=2     k value to get to 5%
        
        # 0.3             50%                 10
        # 0.5             35%                 6        <- default
        # 0.7             25%                 4
        # 0.9             20%                 3
    
        # decay = (1-self.decay_min)*np.exp(self.decay_const*k) + self.decay_min
        
        # default decay = 0.8*e^(-0.5k) + 0.2
        # then allowance used is: allowance * decay
        
        if delete_old == [] or delete_old == False:
            self.delete_old = False
        else:
            self.delete_old = True
            
        self.delete_old = True
        
        # create a move_array list for later use
        self.move_array_list = [[[[0,0,0],[0,0,0]]]]
        
        if opening_book == True:
            # load additional module
            import pickle
            
            self.opening_book = True

            with open('opening_book.data', 'rb') as filehandle:
                # read the data as binary data stream
                storage_list = pickle.load(filehandle)
            
            self.move_array_list = storage_list[0]
            self.eval_array_list = storage_list[1]
            self.board_array_list = storage_list[2]
            self.id_array_list = storage_list[3]
            self.parent_id_list = storage_list[4]
            
        else:
            self.opening_book = False

    def delete(self):
        '''Delete first list in each data structure'''
        
        del self.move_array_list[0]
        del self.board_array_list[0]
        del self.eval_array_list[0]
        del self.id_array_list[0]
        del self.parent_id_list[0]
        del self.all_board_moves[0]
        del self.active_node_array[0]
        del self.active_node_evals[0]
        del self.num_nodes_array[0]
        del self.sorted_hash_array[0]
        del self.sorted_hash_keys[0]
            
    def print_board(self,print_id):
        '''This method prints the board state given'''
        
        board_to_print = self.board_array_list[print_id[0]-self.offset][print_id[1]]
        
        bl.print_board(board_to_print)
    
    def create(self,board_id,move,move_modifier):
        '''This method creates a new board array based on the old board and the
        given move. It evaluates the board and saves the evaluation in the
        eval_array_list. Then, it outputs the index of the new board and
        evaluation - both will have the same index by design'''

        # extract relevant indexes
        board_list_ind = board_id[0] - self.offset    # which list is the board in
        board_item_ind = board_id[1]    # which item is the board in that list
        
        # get a coppy of the old board array
        new_board = copy.deepcopy(self.board_array_list[board_list_ind][board_item_ind])
        
        # find out whos go it is
        if new_board[move[0]][2] == 1:
            white_to_play = False
        elif new_board[move[0]][2] == 2:
            white_to_play = True
        else:
            print('ERROR: self.create has been told to make an invalid move, there is no piece at the desired square')

        # make the move on the board copy
        bl.make_move(new_board,move,move_modifier)
        
        # evaluate the board
        board_eval = bl.eval_board_2(new_board,white_to_play)
        
        # use the new_create function
        data_id = self.new_create(board_id,move,move_modifier,new_board,board_eval)
        
        return data_id
    
    def new_create(self,board_id,move,move_modifier,new_board,board_eval):
        '''This method creates a new board array based on the old board and the
        given move. It evaluates the board and saves the evaluation in the
        eval_array_list. Then, it outputs the index of the new board and
        evaluation - both will have the same index by design'''
        
        # find the hash ID of the board
        hash_id = hash(str(new_board[:100]))
        
        list_num = board_id[0] - self.offset
        
        # define the move array parent id:
        #[[move_num,parent_list,child_item_id],[start_sq,dest_sq,move_modifier]]
        # NB board_id[1]+2 because the parent ID take up to item slots
        parent_id = [[board_id[0],board_id[1],0],
                     [move[0],move[1],move_modifier]]
        
        # check if the board_array_list needs a new column
        if list_num + 2 > len(self.board_array_list):
            
            # create new list entry in board array list, then save
            self.board_array_list.append([])
            self.board_array_list[list_num+1] = [new_board]
            
            # create new list entry in evaluation array list, then save
            self.eval_array_list.append([])
            self.eval_array_list[list_num+1] = [[board_eval]]
            
            # create new list entry in id array list, then save
            self.id_array_list.append([])
            self.id_array_list[list_num+1] = [[hash_id]]
            
            # add the move and evaluation as children of the previous list
            self.eval_array_list[list_num][board_id[1]].append(board_eval)
            self.move_array_list[list_num][board_id[1]].append([move[0],
                                                                   move[1],
                                                                   move_modifier])
            self.id_array_list[list_num][board_id[1]].append(hash_id)
            self.active_node_array[list_num][board_id[1]].append(True)
            
            child_ind = len(self.move_array_list[list_num][board_id[1]])-1
            
            # update the parent id with the child information
            parent_id[0][2] = child_ind
            
            # create new list entry in move array list, then save
            self.move_array_list.append([])
            self.move_array_list[list_num+1] = [parent_id]
            
            # create a new list entry in parent id list, then save
            self.parent_id_list.append([])
            self.parent_id_list[list_num+1] = [[parent_id[0]]]
            # self.parent_id_list[board_list_ind][board_item_ind].append(parent_id)
            
            # create a new list entry in all board moves list, nothing to save
            self.all_board_moves.append([[]])
            
            # create a new list in the active node array
            self.active_node_array.append([[]])
            self.active_node_evals.append([[]])
            
            # add the new list and the append child information
            self.num_nodes_array.append([[0]])
            self.num_nodes_array[list_num][board_id[1]].append(1)
            
            # add new lists for the sorted hash array and keys
            self.sorted_hash_array.append([0,hash_id])
            # self.sorted_hash_array.append([hash_id])
            self.sorted_hash_keys.append([0])
            
        else:    # add the entries to the end of the existing column
        
            # add the evaluation and move as children of the previous list
            self.eval_array_list[list_num][board_id[1]].append(board_eval)
            self.move_array_list[list_num][board_id[1]].append([move[0],
                                                                   move[1],
                                                                   move_modifier])
            self.id_array_list[list_num][board_id[1]].append(hash_id)
            
            self.active_node_array[list_num][board_id[1]].append(True)
            
            self.num_nodes_array[list_num][board_id[1]].append(1)
            
            # save the item that the move is in the previous list
            child_ind = len(self.move_array_list[list_num][board_id[1]])-1
            
            # update the parent id with the child information
            parent_id[0][2] = child_ind
        
            # check if the board already exists in the data structure
            (exists,new_board_id) = self.check_hash_id(board_id,hash_id)
            
            # if the board already exists in the data structure
            if exists:
                
                try:
                    self.parent_id_list[new_board_id[0]-self.offset][new_board_id[1]].append(parent_id[0])
                except:
                    print('new_board_id was',new_board_id)
                    raise ValueError
                    
                return new_board_id
            
            else:   # the new board need to be saved
            
                self.parent_id_list[list_num+1].append([parent_id[0]])
                
                # append the data structure lists with the new data
                self.board_array_list[list_num+1].append(new_board)
                self.eval_array_list[list_num+1].append([board_eval])
                self.move_array_list[list_num+1].append(parent_id)
                self.id_array_list[list_num+1].append([hash_id])
                self.all_board_moves[list_num+1].append([])
                self.active_node_array[list_num+1].append([])
                self.active_node_evals[list_num+1].append([])
                self.num_nodes_array[list_num+1].append([0])
                
                # we need to add the hash information to the correct place
                present,index = self.binary_lookup(hash_id, self.sorted_hash_array[list_num+1])
                
                if present:
                    raise ValueError('new_create should never get here')
                    
                key = len(self.board_array_list[list_num+1]) - 1
                
                self.sorted_hash_array[list_num+1].insert(index,hash_id)
                self.sorted_hash_keys[list_num+1].insert(index-1,key)
        
        # save the data id of the new board
        data_id = [board_id[0]+1,
                   len(self.eval_array_list[list_num+1])-1]
            
        # now return the index where the entries were saved
        return data_id
    
    def check_hash_id(self,board_id,hash_id):
        '''This function checks if the new board already exists in the data
        structure in the current list (not other lists), using the hash ID'''
        
        list_num = board_id[0] - self.offset
        
        # does the hash id exist already?
        present,index = self.binary_lookup(hash_id,self.sorted_hash_array[list_num+1])
        
        if present:
            existing_id = [board_id[0]+1, self.sorted_hash_keys[list_num+1][index-2]]
            return True,existing_id
        else:
            return False,None
    
    def decay(self,k):
        '''This function sets the decay rate for evaluation trimming'''
        #decay = (k//2)+1
        
        decay = (1-self.decay_min)*np.exp(-self.decay_const*k) + self.decay_min
        
        return decay
    
    def width_decay(self,k):
        '''This function sets a new width based on the depth'''
        
        width_drop = (1/2) * self.width
        decay_rate = 0.5
        
        width_reduction = width_drop * (1 - np.exp(-decay_rate*k)) 
        width = self.width - width_reduction
        
        # round this down to an integer value
        width = int(width//1)
        
        return width
    
    def is_root_child(self,node_id):
        '''This method goes upwards through the tree and returns True if the given
        parent leads back to the root of the position, and False if it does not.
        If branching occurs, it returns True if either branch leads to the root'''
        
        list_num = node_id[0] - self.offset
        
        # # is this node active?
        # if self.active_node_array[list_num][node_id[1]][node_id[2]-2] == False:
        #     # if not active, we end the search
        #     return False

        # so long as we haven't finish the tree
        if list_num > 0:
            
            # find the parents of this node
            for parent in self.parent_id_list[list_num][node_id[1]]:
                
                # recursively move upwards
                output = self.is_root_child(parent)
                
                # if we have found the root, we can stop
                if output:
                    break
          
        # if we are at the top of the tree
        elif list_num == 0:
            
            # are we at the root node?
            if node_id[0:2] == self.root_node[0:2]:
                output = True
            
            else:    # this line doesn't lead to the root
                output = False
                
        return output
    
    def binary_lookup(self,item,dictionary):
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
    
    def remove_duplicates(self,mylist):
        '''This method removes duplicates in a given list, whilst maintaining
        the order of the list'''
        
        lookup_list = [0]
        new_list = []
        
        # loop through every item in the list
        for item in mylist:
            
            # hash the item in the list (convert to string first)
            hashed = hash(str(item))
            
            # find if the item is in the lookup_list
            present,index = self.binary_lookup(hashed,lookup_list)
            
            # if the item is not already present
            if not present:
                
                # save this non duplicate in the new list
                new_list.append(item)
                
                # save the hash of this item in the lookup list
                lookup_list.insert(index,hashed)
        
        return new_list
        
    def add_checkmate_nodes(self,id_list):
        '''This method adds checkmate nodes to the list of ids used in the
        global cascade function, also activating the checkmate child'''
        
        i = 0
        
        # what level of the tree are we on? [x][0] should be the same for all ids
        tree_level = id_list[0][0]
        
        # loop through all of the nodes in the reactivate list
        for node in self.reactivate_list:
            
            # check if the node is from a cut out list
            if node[0] - self.offset < 0:
                # in which case cut it out
                self.reactivate_list.pop(i)
                continue
            
            # next, check if this node is at the correct tree level
            if node[0] == tree_level:
                
                # activate the child node
                self.active_node_array[node[0]-self.offset][node[1]][node[2]-2] = True
                
                b_eval = self.eval_array_list[node[0]-self.offset][node[1]]\
                       [node[2]-1]
                     
                # is this position a draw or a checkmate?
                if b_eval != 0.0:
                    
                    # if it is a checkmate, in whose favour is it?
                    if b_eval > 0:
                        b_sign = 1
                    else:
                        b_sign = -1
            
                    # how far from the current move is this mate
                    mate_in = node[0] - self.offset
                    
                    # code this information into the evaluation
                    mate_eval = round(100.1 - (mate_in/1000),4)
                    
                    # adjust to the correct sign
                    mate_eval *= b_sign
                    
                    # save this new evaluation of the checkmate
                    self.eval_array_list[node[0]-self.offset][node[1]][node[2]-1] = mate_eval
                
                # # find the evaluation of the board, save before appending
                # b_eval = self.eval_array_list[node[0]-self.offset][node[1]][node[2]-1]
                
                # add the parent to the id list
                id_list.append(node)
            
            # increment and loop to next element
            i += 1
            
        return id_list
    
    def reactivate_cascade(self,node_id):
        '''This method checks if a node has any active children, and if so it
        reactivates the parent, then it repeats this check for the parent and
        so on, recursively'''
        
        list_num = node_id[0] - self.offset
        
        # so long as we haven't finish the tree
        if list_num > 0:
            
            node_still_active = False
            
            # are there any active nodes here?
            for f in self.active_node_array[list_num][node_id[1]]:
                # if a node is active, the cascade is finished
                if f == True:
                    node_still_active = True
                    break
                    
            # were any of the children active?
            if node_still_active == False:
                # if not, we do not need to reactivate this node
                return
            
            # hence, this node has no active children
            
            # loop through the parents of this node
            for parent in self.parent_id_list[list_num][node_id[1]]:
                
                # is the node set to deactive?
                if (self.active_node_array[parent[0]-self.offset][parent[1]]\
                    [parent[2]-2]) == False:
                    
                    # in which case we need to reactivate it
                    self.active_node_array[parent[0]-self.offset][parent[1]]\
                    [parent[2]-2] = True
                    
                    # recursively check whether more nodes need to be reactivated
                    self.reactivate_cascade(parent)
        
        # we have finished the tree, or run out of nodes to set active
        return
    
    def deactivate_cascade(self,node_id):
        '''This method checks if a node has no active children, and if not it
        deactivates the parent, then it repeats this check for the parent and
        so on, recursively'''
        
        list_num = node_id[0] - self.offset
        
        # so long as we haven't finish the tree
        if list_num > 0:
            
            # are there any active nodes here?
            for f in self.active_node_array[list_num][node_id[1]]:
                # if a node is active, the cascade is finished
                if f == True:
                    return
                
            # hence, this node has no active children
            
            # loop through the parents of this node
            for parent in self.parent_id_list[list_num][node_id[1]]:
                
                # set the child entry for this current node in the parent to false
                self.active_node_array[parent[0]-self.offset][parent[1]]\
                    [parent[2]-2] = False
                
                # recursively check if the parents contain any active nodes
                self.deactivate_cascade(parent)
        
        # we have finished the tree, or run out of inactive nodes
        return
        
    
    def deactivate_nodes(self,set_to=False,skip_end=0):
        '''This method sets every node to inactive'''
        
        # go through every single list in the active node array
        for i in range(len(self.active_node_array)-skip_end):
            for j in range(len(self.active_node_array[i])):
                
                # replace this list with a list of False
                elements = len(self.active_node_array[i][j])   
                self.active_node_array[i][j] = [set_to for k in range(elements)]
                
        # TESTING AN IDEA
        # lets reset all of the node counts to zero as well
        # go through every single list in the active node array
        for i in range(len(self.num_nodes_array)-skip_end):
            for j in range(len(self.num_nodes_array[i])):
                
                # replace this list with a list of False
                elements = len(self.num_nodes_array[i][j])   
                self.num_nodes_array[i][j] = [0 for k in range(elements)]
                
    def validate_active(self, node_id):
        '''This method checks to confirm that a given node is on the end of an
        active branch. In the event of branching, all branches are checked'''
        
        list_num = node_id[0] - self.offset
        
        is_active = False       # starting assumption
        
        # so long as we haven't finish the tree
        if list_num > 0:
            
            # loop through the parents of this node
            for parent in self.parent_id_list[list_num][node_id[1]]:
                
                node_active = False
                
                # is this parent active
                for f in self.active_node_array[parent[0]-self.offset]\
                    [parent[1]]:
                    if f == True:
                        node_active = True
                        break
                    
                # if the parent is not active
                if node_active == False:
                    continue
                
                # recursively check if the parents are active
                is_active = self.validate_active(parent)
                
                # if we get a true, no need to check any other branches
                if is_active == True:
                    break
                
        elif list_num == 0:
            
            # we are finished
            return True
            
            node_active = False
        
        # we have finished the branch, or run out of active nodes
        return is_active
            
    def global_cascade(self,id_list,white_to_play,d):
        '''This method performs cascade on all boards in a global manner, only
        moving up the cascade when all of the boards at that level have been
        updated'''
        
        cascade_player = white_to_play
        
        # loop through every level of the search tree
        for k in range(d+1):
            
            if cascade_player:
                sign = 1
            else:
                sign = -1
                
            next_id_list = []
            already_checked = []
            
            # errors are emerging where the id_list goes to zero
            if len(id_list) == 0:
                print('In global cascade, the id_list went to zero')
                test = 4+4
                test = 4
                return
            
            # add any checkmate nodes to the list
            id_list = self.add_checkmate_nodes(id_list)
            
            # remove duplicates from the id list
            id_list = self.remove_duplicates(id_list)
            
            # loop through all the ids in our list
            for id_item in id_list:
                
                # # don't bother rechecking already done ids
                # if id_item[0:2] in already_checked:
                #     continue
                # else:
                #     already_checked.append(id_item[0:2])
                
                list_num = id_item[0] - self.offset
                    
                # default starting value thats guaranteed to be beaten
                max_eval = 1000 * sign * -1
                
                num_active_nodes = 0
                
                # loop through every child evaluation for this node
                for i in range(len(self.eval_array_list[list_num]\
                                   [id_item[1]][1:])):
                    
                    # is the evaluation from an active downstream node
                    if self.active_node_array[list_num][id_item[1]][i]:
                        
                        next_eval = self.eval_array_list[list_num][id_item[1]][i+1]
                        
                        num_active_nodes += self.num_nodes_array[list_num]\
                            [id_item[1]][i+1]
                        
                        # is this evaluation better than the current best
                        if max_eval*sign < sign*next_eval:
                            
                            # if it is better, it is our new best
                            max_eval = next_eval
                    
                    else:    # the node is not active
                        
                        # hence there are no active downstream nodes
                        self.num_nodes_array[list_num][id_item[1]][i+1] = 0
                
                # catch cases where there were no active nodes
                if num_active_nodes == 0:
                    # this entire node is dead, zero downstream active nodes
                    self.num_nodes_array[list_num][id_item[1]][0] = 0
                    # MAYBE PUT DEACTIVATE CASCADE HERE
                    # best to wait until self.prune since then the whole
                    # tree is finished
                    # however, im not sure this will ever trigger since we
                    # are only deactivating nodes now??
                    #self.deactivate_cascade(id_item)
                    continue
                
                # save the number of active nodes
                self.num_nodes_array[list_num][id_item[1]][0] = num_active_nodes
                        
                # now we have the best evaluation for this board
                
                # save this best evaluation
                self.eval_array_list[list_num][id_item[1]][0] = max_eval
                
                # TESTING THIS CODE - it goes with for k in range(d+1):
                # end here if on the final loop
                if k == d:
                    break
                    
                # save the parent information for the next loop and overwrite
                # previous evaluations
                for parent in self.parent_id_list[list_num][id_item[1]]:
                    
                    # does this parent lead to the root?
                    # does this parent lead to the root position?
                    # GET RID OF IS_ROOT_CHILD, note how I skipped it with 'or True'
                    if (len(self.parent_id_list[id_item[0]-self.offset][id_item[1]])\
                        == 1 or self.is_root_child(parent) or True):
                        
                        # put the parent into the next id list
                        # next_id_list.append(parent[:2])
                        next_id_list.append(parent)
                        
                        # if the node is not yet 
                        # AGAIN note the skip with 'or True'
                        if self.active_node_array[parent[0]-self.offset]\
                            [parent[1]][parent[2]-2] == False or True:
                                
                                # update the eval array for the parent child
                                self.eval_array_list[parent[0]-self.offset]\
                                    [parent[1]][parent[2]-1] = max_eval
                                
                                # # WE DONT NEED THIS IF WE ONLY DEACTIVATE NODES
                                # # set the node to active
                                # self.active_node_array[parent[0]-self.offset]\
                                #     [parent[1]][parent[2]-2] = True
                                    
                                # update the parent with the number of downstream nodes
                                self.num_nodes_array[parent[0]-self.offset]\
                                    [parent[1]][parent[2]-1] = self.num_nodes_array\
                                        [list_num][id_item[1]][0]
                                    
                        else:    # the node is already active
                        
                            # THIS CODE IS NEVER REACHED, IT NEEDS TO GO
                        
                            print('hereeee in global_cascade')
                        
                            # we only update if it is a better move for the other player
                            if max_eval*sign*-1 > -1*sign*self.eval_array_list\
                                [parent[0]-self.offset][parent[1]][parent[2]-1]:
                                    
                                print('better eval')
                                    
                                # update the eval array
                                self.eval_array_list[parent[0]-self.offset]\
                                    [parent[1]][parent[2]-1] = max_eval
                                    
                        break
                            
                # now we have finished with this id_item
            
            # now we have finished with this id_list
            id_list = next_id_list
            
            cascade_player = not cascade_player
            
            # now we loop to next k
            
    def test_legal_moves(self,list_one,list_two,board_id):
        '''This method checks whether two lists are exactly the same, to see if
        total legal moves is working correctly'''
        
        # first, convert each list to a hashlookup_list = [0]
        new_pair = [[], []]
        hash_list = [[0],[0]]
        old_pair = [list_one, list_two]
        
        for i in range(2):
            for item in old_pair[i]:
                
                # hash the item in the list
                hashed = hash(str(item))
                
                # sort these hashes into a new list
                present,index = self.binary_lookup(hashed,hash_list[i])
                
                # if the item is not already present
                if not present:
                    
                    # save this non duplicate item in the new list
                    new_pair[i].insert(index-1,item)
                    
                    # save this non duplicate hash in the lookup list
                    hash_list[i].insert(index,hashed)
                    
                else:    # we do not expect any duplicates
                
                    print('Found a duplicate in test legal moves')
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
                
        # now see whether the legal moves are the same
        if wrong > 0:
            
            # mistakes were made
            # print the board
            bl.print_board(self.board_array_list[board_id[0]-self.offset]\
                           [board_id[1]],False)
            print('Length new function =',len(list_one),'Length old = ',len(list_two))
            print('The number of errors was ',wrong)
            print('The sorted lists are:')
            print('New: ',new_pair[0])
            print('Old: ',new_pair[1])
        
        return
    
    def list_compare(self,list_one,list_two):
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
                present,index = self.binary_lookup(hashed,hash_list[i])
                
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
    
    def global_branch_limits(self,next_id_list,white_to_play,white_comp,d):
        '''This method prepares to trim the branches of the search tree according
        to preset limits on the branching factor at each node'''
        
        # what is the maximum allowable number of branches?
        _,max_branches = self.set_branch_limits(0,d,True)
        
        print('The max branches is set as', max_branches)
        
        # start at the beginning
        node_id = self.root_node
        
        cascade_player = white_comp
        
        # prepare to recursively traverse the tree
        deactivated = self.branch_limit(node_id,max_branches,cascade_player,d)
        
        # now remove any deactivated nodes from the next_id_list
        
        lookup_list = [0]
        new_list = []
        
        # save a hash of each deactivated node
        for item in deactivated:
            
            # hash the item in the list (convert to string first)
            hashed = hash(str(item))
            
            # find if the item is in the lookup_list
            present,index = self.binary_lookup(hashed,lookup_list)
            
            # if the item is not already present
            if not present:
                
                # # save this non duplicate in the new list
                # new_list.append(item)
                
                # save the hash of this item in the lookup list
                lookup_list.insert(index,hashed)
                
        # now go through the new_ids and remove duplicates or deactivated
        for item in next_id_list:
            
            # hash the item in the list (convert to string first)
            hashed = hash(str(item[0:2]))
            
            # find if the item is in the lookup_list
            present,index = self.binary_lookup(hashed,lookup_list)
            
            # if the item is not already present
            if not present:
                
                # TESTING
                # perform extra validation
                is_active = self.validate_active(item)
                
                # only add if we can validate that it is active
                if is_active:
                    # save this non duplicate in the new list
                    new_list.append(item)
                
                # save the hash of this item in the lookup list
                lookup_list.insert(index,hashed)
                
        # testing: lets only add the best nodes
        # we enfore the max branch limit very hard, sorting the new ids by how
        # good their evaluation is, and chopping off the bottom
        if white_to_play:
            # these signs need to be backwards as binary lookup puts the lowest
            # numbers first, hence we need lowest=best
            sign = -1
        else:
            sign = 1
        final_list = [0]
        final_dict = [0]
        for item in new_list:
            # find where the evaluation puts this item in the list
            present,index = self.binary_lookup(item[2]*sign,final_dict)
            
            # put the item into the sorted list
            final_list.insert(index,item)
            final_dict.insert(index,item[2])
        # now, trim the final list in line with branch limits
        final_list = final_list[1:int(max_branches)+1]
        new_list = final_list[:]
        # end of testing code
        
        return new_list
    
    def set_branch_limits(self,k,d,top=False):
        '''This method determines the minimum branch number and branching
        gradient at any particular node'''
        
        width = self.width
        
        speed_factor = self.speed_factor
        
        if top == True:
            min_prop = width * speed_factor
            min_grow = 1.1
            growth_decay = width/2.5
        else:
            min_prop = (1-speed_factor)
            min_grow = 1.1
            growth_decay = width/2.5
            
        # what is our starting minimum for a node
        minimum = ((min_prop*width)//1)
    
        # what level of the tree are we on
        level = (d-k-1)    
        
        # loop through, a higher level gives a bigger minimum
        for x in range(level):
            
            growth = width - (x*growth_decay)
            
            # make sure growth doesn't drop below a certain level
            if growth < min_grow:
                growth = min_grow
            
            # scale up the minimum each level
            minimum *= growth
            
        gradient = 5 + 5*level
        
        # TESTING a booster shot
        # this seems to be working well
        if top == True:
            minimum += (0.5*width**2)
            
        return gradient,minimum//1
    
    def kill_branch(self,node_id,d):
        '''This method deactivates a branch and all of its children using
        recursion until it gets to the end of the tree'''
        
        list_num = node_id[0] - self.offset
        
        deactivated_nodes = []
        
        # loop through all the child nodes of this branch
        for i in range(len(self.active_node_array[list_num][node_id[1]])):
            
            # is this node active
            if self.active_node_array[list_num][node_id[1]][i] == False:
                continue
            
            # deactivate the node
            self.active_node_array[list_num][node_id[1]][i] = False
            
            # find the child of this node
            exists,data_id = self.check_hash_id(node_id,self.id_array_list[list_num]\
                                         [node_id[1]][i+1])
                
            if exists == False:
                print('the node doesnt exist - this should NOT happen in kill_branch')
                continue
            
            # how many parents does this child node have
            if len(self.parent_id_list[data_id[0]-self.offset][data_id[1]]) > 1:
                
                keep_going = True
                
                # check if any other parents are active
                for parent in self.parent_id_list[data_id[0]-self.offset][data_id[1]]:
                    
                    # if they are active, we shouldn't keep going deactivating
                    if self.active_node_array[parent[0]-self.offset][parent[1]]\
                        [parent[2]-2] == True:
                        keep_going = False
                        
                # keep_going = True
                
                # if this child has another active parent, we shouldnt deactivate it
                if keep_going == False:
                    continue

            # are we now at the end of the tree?
            if list_num == d:
                
                # save this leaf as deactivated
                deactivated_nodes.append(data_id)
                
                #self.active_node_array[data_id[0]-self.offset][data_id[1]] = [False]
                
            else:    # we need to go deeper
                
                # deactivate the child
                newly_deactivated = self.kill_branch(data_id,d)
            
                # add to the list
                deactivated_nodes += newly_deactivated
                
        return deactivated_nodes
                
    def branch_limit(self,node_id,allowed_branches,cascade_player,d):
        '''This method applies a branch limit to a specific node, and may call
        itself recursively to apply branch limits at downstream nodes in order
        to satisfy the limit at this node'''
        
        list_num = node_id[0] - self.offset
        
        if cascade_player:
            c_sign = 1
        else:
            c_sign = -1
        
        num_branches = self.num_nodes_array[list_num][node_id[1]][0]
        
        # if this node is under the branch limit already
        if num_branches <= allowed_branches and False:
            # TESTING CODE: inserted false above
            return []
        
        # if this node is at the end of the tree
        elif list_num == d:
            # print('yes this works')
            # changing this number +1 or -1 causes exponential growth
            # since deactivation no longer works properly
            return []
        
        # if we need to reduce the number of branches at this node
        else:
            
            # what is the best evaluation at this node
            best_eval = self.eval_array_list[list_num][node_id[1]][0]
            
            info = []
            total_loss = 0.0
            num_active = 0
            
            # allocate allowed branches to each active downstream node
            for i in range(len(self.active_node_array[list_num][node_id[1]])):
                
                # is this node active
                if self.active_node_array[list_num][node_id[1]][i] == False:
                    # if not, skip it
                    continue
                
                num_active += 1
                
                # what is the evaluation of this node
                node_eval = self.eval_array_list[list_num][node_id[1]][i+1]

                # work out how much worse this node is than the best one
                loss = (best_eval * c_sign) - (c_sign * node_eval)
                
                # keep a running total of the loss
                total_loss += loss
            
                # save this information (i+1 is index in id_array_list/eval_array_list
                new_elem = [loss, i+1]
            
                # put it in a sorted list
                if num_active == 1:
                    info.append(new_elem)
                else:
                    for j in range(len(info)):
                        if loss < info[j][0]:
                            info.insert(j,new_elem)
                            break
                    if j == len(info)-1:
                        info.append(new_elem)
            
            # if we come across a dead branch
            if num_active == 0:
                # TESTING CODE
                pruned_nodes = []
                # go through the list of nodes to deactivate
                for item in info:
                
                    # which is the child of this node
                    exists,kill_id = self.check_hash_id(node_id,self.id_array_list\
                                                 [list_num][node_id[1]][item[1]])
                        
                    # deactivate this nodes child entry in the parent list
                    self.active_node_array[list_num][node_id[1]][item[1]-1] = False
                      
                    # kill it!
                    pruned_nodes += self.kill_branch(kill_id,d)
                
                # once we have double checked they are all dead
                return pruned_nodes
                # END OF TESTING CODE
                # before it was just: return []
 
            # determine the branch limiting factors
            grad_b,min_b = self.set_branch_limits(list_num,d,False)
            
            deactivate_list = []
            
            # loop until the branching criteria is satisfied
            while True:
                
                # find the greatest loss
                max_loss = info[-1][0]
                
                # find the average loss per branch
                avg_loss = total_loss / num_active
                
                # find num branches for this loss
                avg_branches = min_b + (grad_b * (max_loss - avg_loss))
                
                # find the total number of branches
                total_branches = avg_branches * num_active
                            
                # have we satisfied our requirement?
                if total_branches <= allowed_branches:
                    break
                else:    # kill the node with the largest loss
                    total_loss -= max_loss
                    num_active -= 1
                    # the killed node needs to be deactivated
                    deactivate_list.append(info.pop())
                    # are we out of nodes?
                    if len(info) == 0:
                        break
                    
            # deactivate nodes
            pruned_nodes = []
                    
            # find out the scaling factor to bump up to maximum branches
            scale = allowed_branches / total_branches
            
            # go through the list of nodes to deactivate
            for item in deactivate_list:
                
                # which is the child of this node
                exists,kill_id = self.check_hash_id(node_id,self.id_array_list\
                                             [list_num][node_id[1]][item[1]])
                    
                # deactivate this nodes child entry in the parent list
                self.active_node_array[list_num][node_id[1]][item[1]-1] = False
                  
                # kill it!
                pruned_nodes += self.kill_branch(kill_id,d)
            
            # branch_total = 0
            
            # loop through the remaining nodes to allocate the branches
            for item in info:
                
                # how many branches is this node asssigned
                branches_assigned = (scale * (min_b + grad_b * (max_loss - item[0]))) // 1
                
                # which node is it
                exists,data_id = self.check_hash_id(node_id,self.id_array_list\
                                                    [list_num][node_id[1]][item[1]])
                
                # recursively call this function
                inactive_set = self.branch_limit(data_id,branches_assigned,
                                                 not cascade_player,d)
                
                # add the inactive nodes to the list
                pruned_nodes += inactive_set
                
                # branch_total += branches_assigned
                # print('Node',item[1],'gets',branches_assigned,'branches')
                
            # print('The total branches was',branch_total,'(',allowed_branches,')')
            # print(info)
                
            return pruned_nodes
            
    
    def global_prune(self,next_id_list,id_nums,white_to_play,d):
        '''This method prunes the search tree using a cutoff value, which gets
        smaller as the depth increases to try to counteract the branching'''
        
        # set up parameters for prune function
        old_eval = None
        kmin = 0
        kmax = d
        
        # prepare to traverse the id list
        lower_bound = 0
        new_next_list = []
        lookup_list = [0]     # for hash ids of each item, to prevent duplicates
        
        # loop through each set of new ids
        for set_num in id_nums:
            
            if set_num == 0:
                continue
            
            # extract the new ids for this branch
            id_set = next_id_list[lower_bound:lower_bound+set_num]
            
            # no_dupes_set = []
            
            # # go through the new_ids to remove duplicates
            # for item in id_set:
                
            #     # hash the item in the list (convert to string first)
            #     hashed = hash(str(item))
                
            #     # find if the item is in the lookup_list
            #     present,index = self.binary_lookup(hashed,lookup_list)
            
            #     # if the item is not already present
            #     if not present:
                    
            #         # save this non duplicate in the new list
            #         no_dupes_set.append(item)
                    
            #         # save the hash of this item in the lookup list
            #         lookup_list.insert(index,hashed)
                    
            # id_set = no_dupes_set
            
            # prune these new ids by checking if their evaluations are good enough
            new_ids = self.prune(id_set,old_eval,white_to_play,kmin,kmax)
            
            # only add these new_ids if they are not duplicates
            for item in new_ids:
                
                # hash the item in the list
                hashed = hash(str(item))
                
                # find if the item is in the lookuplist
                present,index = self.binary_lookup(hashed,lookup_list)
                
                # if the item is not present
                if not present:
                    
                    # save this non duplicate in the new list
                    new_next_list.append(item)
                    
                    # save the hash of this item in the lookup list
                    lookup_list.insert(index,hashed)

            # # save the output (uncomment this only if we check duplicates before pruning)
            # new_next_list += new_ids
            
            # update for next loop
            lower_bound += set_num
            
        # finished pruning
        return new_next_list
    
    def prune(self,new_ids,old_eval,white_to_play,kmin,kmax,parent_id=[],base_parent=[]):
        '''This method performs cascade on all boards in a global manner, only
        moving up the cascade when all of the boards at that level have been
        updated'''
        
        cascade_player = white_to_play
        
        # check if we are on a recursive loop
        if parent_id != []:
            next_id = parent_id
        else:
            next_id = parent_id
        
        # check if we have been given nothing
        if len(new_ids) == 0:
            return []

        # now loop through all the specified levels of the game tree
        for k in range(kmin,kmax):
            
            if cascade_player:
                sign = 1
            else:
                sign = -1
            
            if k == 0:

                # next_id = new_ids[0][0:2]
                list_num = new_ids[0][0] - self.offset
                max_eval = new_ids[0][2]
                
                # if there is only one parent, it must be shared by all new_ids
                if len(self.parent_id_list[list_num][new_ids[0][1]]) == 1:
                    
                    base_parent = self.parent_id_list[list_num][new_ids[0][1]][0]
                                                                
                else:    # if the first new id has multiple parents
                    
                    # save the list of parents
                    poss_parents = self.parent_id_list[list_num][new_ids[0][1]][:]
                    
                    # go through all the other new_ids
                    for item in new_ids:
                        # check which parents they share with the first new_id
                        for possible in poss_parents:
                            # if a parent is not shared, it cannot be the base parent
                            if possible not in self.parent_id_list[list_num]\
                                [item[1]]:
                                poss_parents.remove(possible)
                        # have we narrowed it down to one parent?
                        if len(poss_parents) == 1:
                            break
                    
                    # any parents shared by all are used as base
                    if len(poss_parents) == 1:
                        base_parent = poss_parents[0]
                        
                    else:    # in the rare case we have multiple shared parents
                        # we still take only one base parent
                        base_parent = poss_parents[0]
                         
                        # FOR TESTING - IS THIS A COMMON OCCURANCE??
                        # so far it happens mostly when len(new_ids)==1 (so a forcing
                        # line) or len(new_ids)==2, quite forcing
                        # print('Found multiple base parents in self.prune, new_id[0] is',
                        #       new_ids[0][0:2],'base parent used was',base_parent[0:2])
                        # print('The parent id list is',self.parent_id_list[list_num][new_ids[0][1]])
                        
                        # but we put this base parent at the end of all the new_ids
                        # parent lists, so if we repeat this with other sets
                        # of new_ids, the other parents will be selected
                        for item in new_ids:
                            for c,entry in enumerate(self.parent_id_list[list_num][item[1]]):
                                if entry[0:2] == base_parent[0:2]:
                                    break
                            # c is the index of the base parent in the list
                            # put the parent at the end of the parent_id_list
                            self.parent_id_list[list_num][item[1]].pop(c)
                            self.parent_id_list[list_num][item[1]].append(base_parent)
                 
                # next, cut out any terrible evaluations in new_ids
                
                # cutoff is defined by best new_id evaluation
                leaf_cutoff = new_ids[0][2]*sign - (self.allowance*self.decay(k))
                 
                # loop through the rest of new_ids
                for x in range(1,len(new_ids)):
                    # if the evaluation drops below cutoff, remove the rest of new_ids
                    if leaf_cutoff > new_ids[x][2]*sign:
                        
                        # now deactivate the nodes before we cut them
                        for y in range(x,len(new_ids)):
                            # use the base parent
                            self.active_node_array[base_parent[0]-self.offset]\
                                [base_parent[1]][base_parent[2]-2] = False
                                
                            # for parent in self.parent_id_list[list_num][new_ids[y][1]]:
                            #     self.active_node_array[parent[0]-self.offset]\
                            #         [parent[1]][parent[2]-2] = False
                          
                        # now cut out the nodes that have dropped below cutoff
                        new_ids = new_ids[:x]
                        break     
            
            else:    # not the first loop
            
                list_num = next_id[0] - self.offset
                
                # the best evaluation at this node should already be stored at
                # [0] in eval_array_list due to global cascade
                
                # check whether the node is dead
                if self.num_nodes_array[list_num][next_id[1]][0] == 0:
                    # check whether upstream nodes are similarly dead
                    self.deactivate_cascade(next_id)
                    new_ids = []
                    return new_ids
                
                # what is the best evaluation at this position (found by global_cascade)
                max_eval = self.eval_array_list[list_num][next_id[1]][0]
                
                # # determine the best evaluation at this node
                # max_eval = 1000 * sign * -1
                
                # # loop through every child evaluation for this node
                # for i in range(len(self.eval_array_list[list_num]\
                #                     [next_id[1]][1:])):
                    
                #     # is the evaluation from an active downstream node
                #     # or is it a checkmate
                #     if self.active_node_array[list_num][next_id[1]][i]:
                #         #or abs(self.eval_array_list[list_num][next_id[1]][i+1]) == 100.1):
                        
                #         # OKAY so below it orginally had [i]
                #         # but the first element in the eval_array_list is an id eval
                #         # so surely it should be [i+1]
                #         # however i swear i changed one of these things from i+1 to i
                #         # so maybe it is now correct and im repeating my past mistake
                        
                        
                #         next_eval = self.eval_array_list[list_num][next_id[1]][i]
                        
                #         # is this evaluation better than the current best
                #         if max_eval*sign < sign*next_eval:
                            
                #             # if it is better, it is our new best
                #             max_eval = next_eval
                                  
                # # finally, catch cases where max eval has not changed
                # if max_eval == 1000 * sign * -1:
                #     new_ids = []
                #     return new_ids
                
                # this is the pruning criteria
                branch_cutoff = max_eval*sign - (self.allowance*self.decay(k))
                
                # testing a more thorough pruning method
                # loop through the rest of new_ids
                for x in range(0,len(new_ids)):
                    # if the evaluation drops below cutoff, remove the rest of new_ids
                    if branch_cutoff > new_ids[x][2]*sign:
                        
                        # now deactivate the nodes before we cut them
                        for y in range(x,len(new_ids)):
                            # use the base_id as the parent
                            self.active_node_array[base_parent[0]-self.offset]\
                                [base_parent[1]][base_parent[2]-2] = False
                            
                            # for parent in self.parent_id_list[list_num][new_ids[y][1]]:
                            #     self.active_node_array[parent[0]-self.offset]\
                            #         [parent[1]][parent[2]-2] = False
                        
                        # cut out the nodes that have dropped below cutoff
                        new_ids = new_ids[:x]
                        break
                    
                # OLD PRUNING WHERE ITS ALL OR NOTHING
                # # if our downstream branch drops below cutoff
                # if old_eval*sign < branch_cutoff:
                    
                #     # end exploration of the downstream branch
                #     new_ids = []
                #     return new_ids
                    
            # now we have finished pruning at this k
                
            # if we have pruned all of the ids
            if len(new_ids) == 0:
                # check if this node is dead (it should be) then check upstream
                self.deactivate_cascade(base_parent)
                
                return new_ids
                    
                    # # deactivate the parent node, and any upstream if needed
                    # for parent in self.parent_id_list[list_num][next_id[1]]:
                        
                    #     self.deactivate_cascade(parent)
                    
                    # return new_ids
                
            # now we prepare to jump up to the next branch
            
            # if we are on the first loop, we have already found the base parent
            if k == 0:
                parent_id = base_parent[0:2]
 
            # if not on the first loop, is there only one parent
            elif len(self.parent_id_list[list_num][next_id[1]]) == 1:
                   
                   # only one parent id
                   parent_id = self.move_array_list[list_num][next_id[1]][0]
                   
            else:    # we have to loop through every parent
                
                potential_ids = []
            
                for parent_id in self.parent_id_list[list_num][next_id[1]]:
                    
                    # only loop if the parent is active
                    if self.active_node_array[parent_id[0]-self.offset]\
                        [parent_id[1]][parent_id[2]-2] == True:
                    
                        # loop recursively (and pray to god it works)
                        ids = self.prune(new_ids,max_eval,not cascade_player,k+1,
                                         kmax,parent_id,base_parent)
                        
                        potential_ids.append(ids)
                    
                # now we have finished looping recursively, so we are done
                max_len = 0
                
                # choose the least trimmed set of ids, just in case
                for ids in potential_ids:
                    if len(ids) > max_len:
                        max_len = len(ids)
                        
                new_ids = new_ids[:max_len]
                
                # now reactivate the leftover nodes (in case they were deactivated)
                for y in range(max_len):
                    # use the base_id as the parent
                    self.active_node_array[base_parent[0]-self.offset]\
                        [base_parent[1]][base_parent[2]-2] = True
                        
                # check if we need to propogate reactivation
                self.reactivate_cascade(base_parent)
                                
                    # try:
                    #     for parent in self.parent_id_list[list_num][new_ids[y][1]]: 
            
                    #         # reactivate
                    #         self.active_node_array[parent[0]-self.offset]\
                    #             [parent[1]][parent[2]-2] = True
                            
                    #         self.reactivate_cascade(parent)
                    # except:
                    #     test = 4+4
                    #     test =4*4
                
                return new_ids
            
            # update the evaluation upstream in the cascade
            # NB parent_id[2]-1 since locations in move_array and
            # eval_array are one offset!
            
            # save the best downstream evaluation for the next loop
            old_eval = max_eval
            
            # update the cascade_id for the next loop
            next_id = parent_id
            
            # change the player as we retreat a depth
            cascade_player = not cascade_player
            
            # advance to the next loop
            
        # we have finished cascading and trimming, now return what's left
        return new_ids
    
    def cut_branches(self,next_id_list,id_nums,white_to_play,
                         white_comp,d,method):
        '''This method handles cutting down the search tree, offering various
        methods for doing so, co-ordinating with available time'''
        
        # cut branches using one of the two methods
        #next_id_list = self.global_branch_limits(next_id_list,white_comp,d)
        next_id_list = self.global_prune(next_id_list,id_nums,white_to_play,d)
        
        pass
                        
    def best_moves_plus_2(self,data_id,white_to_play,best_num):
        '''This function is the same as the original best moves but aims to save
        time by only reevaluating the pieces that are in view of the piece that
        has moved and actually changed things'''
        
        list_num = data_id[0] - self.offset
        
        old_board = self.board_array_list[list_num][data_id[1]]
        
        checked_already = False
        mating_move = False
        
        # check if this board has already been evaluated
        if self.all_board_moves[list_num][data_id[1]] != []:
            
            checked_already = True
            
            # then we have already evaluated this board
            
            # copy out data of all the moves into considerations
            considerations = [x[:] for x in self.all_board_moves[list_num][data_id[1]]]
            
            # loop through the moves we found and update the evaluations of the
            # moves we took further last time, and now know more about
            for i, evaled in enumerate(self.eval_array_list[list_num][data_id[1]][1:]):
                
                considerations[i][2] = evaled
                
            # finally, sort the all board moves so the best are first
            if not white_to_play:
                considerations = sorted(considerations, 
                                        key=lambda x: x[2])
            else:
                considerations = sorted(considerations, 
                                        key=lambda x: x[2], reverse=True)
                
            # how many moves are on offer
            if len(considerations) < best_num:
                num = len(considerations)
            else:
                num = best_num
                
            board_list = []
                
            # reconstruct the boards for the best moves
            for j in range(num):
                
                start_sq = considerations[j][0]
                dest_sq = considerations[j][1]
                new_eval = considerations[j][2]
                move_mod = considerations[j][3]
                
                # get a copy of the board
                new_board = [x[:] for x in old_board]
    
                # make the move on the board copy
                bl.make_move(new_board,[start_sq,dest_sq],move_mod)
                
                # save this board in the board list
                board_list.append(new_board)
                
                
        else:    # this is a new board to evaluate
        
            # CURRENTLY DURING TESTING THIS IS OVERWRITTEN DIRECTLY BELOW
            old_eval =  self.eval_array_list[list_num][data_id[1]][0]
            
            # get a list of every legal move in the position
            (list_of_legal_moves,data_array,
             old_eval,outcome) = bl.total_legal_moves_plus(old_board,white_to_play)
            
            # TESTING CHECK THE LEGAL MOVE FUNCTION-----------------------
            if self.mode == "testing":
  
                list_one = list_of_legal_moves
                list_two = bl.total_legal_moves_adjusted(old_board,2-white_to_play)
                
                self.test_legal_moves(list_one,list_two,data_id)
                
                # now lets test the cpp version
                cpp_board = cpc.board_to_cpp(old_board)
                tlm_struct = bf.total_legal_moves(cpp_board,white_to_play)
                legal_moves_cpp = cpc.cpp_tlm_to_py(tlm_struct)
                
                is_correct = self.list_compare(list_one,legal_moves_cpp)
                
                if not is_correct:
                    print("The cpp total legal moves is not the same as python")
                    print("List one is python (correct)")
                    print("List two is cpp")
                    bl.print_board(old_board,False)
                    
            #------------------------------------------------------------------
            
            # check if the position is a three fold repition or 50 moves without
            # capture using a list of confirmed positions (hash codes) of the
            # match, then change outcome to 0. Could also put this logic in total_
            # legal_moves_plus
            
            # check what the board outcome is
            if outcome != 0:
                
                # is it checkmate
                # THIS CODE IS NEVER REACHED!!!!! ONLY DRAW IS REACHED
                # maybe that's fine, this is a failsafe
                if outcome == 1:
                    # who has been checkmated
                    if white_to_play:
                        outcome_eval = -100.1
                    else:
                        outcome_eval = 100.1
                
                # is it a draw
                elif outcome == 2:
                    outcome_eval = 0.0
                    
                # now save this board evaluation
                self.eval_array_list[list_num][data_id[1]][0] = outcome_eval
                
                # next update the board evaluation in the parent
                for parent in self.parent_id_list[list_num][data_id[1]]:
                    
                    # overwrite parent evalution
                    self.eval_array_list[parent[0]-self.offset][parent[1]]\
                        [parent[2]-1] = outcome_eval
                
                    # finally, we want this node to remain active
                    self.reactivate_list.append(parent)
                    
                    # this doesn't work since we don't have a three element id
                    # we don't have the child id
                    # self.reactivate_list.append(data_id)
                    
                    # new_ids = [[data_id[0], data_id[1], outcome_eval]]
            
                # return True,new_ids
                return True,[]
            
            
            # # if there are no legal moves, it is checkmate
            # if list_of_legal_moves == []:
                
            #     print('best moves just found a checkmate - is that supposed to happen?')   
            #     return True # end the function, there is nothing to be done
            
            # find out how many moves are on offer
            if len(list_of_legal_moves) < best_num:
                num = len(list_of_legal_moves)
            else:
                num = best_num
            
            # create empty move shell
            if white_to_play:
                # considerations = [[0,0,-1000.0,0] * 1 for p in range(num)]
                # board_list = [[] * 1 for q in range(num)]
                considerations = [[0,0,-1000.0, 0]]
                board_list = [[]]
                sign = 1
            else:
                # considerations = [[0,0,1000.0,0] * 1 for p in range(num)]
                # board_list = [[] * 1 for q in range(num)]
                considerations = [[0,0,1000.0, 0]]
                board_list = [[]]
                sign = -1
            
            # LOGIC TO SORT MOVES INTO PRIORITY LIST?
            # perhaps trim list_of_legal_moves to only include important moves
            # best num is currently not used
            
            # it would be cool to loop through the pieces of the board, just like
            # in eval_board_2 and piece_eval and extract all the legal moves at the
            # same time, and also use that opportunity to sort moves into check,
            # capture etc. Then, after checking all of those high priority cases,
            # we could check moves for pieces underperforming based on their
            # piece_eval, perhaps once we have checked a bunch of moves that result
            # in poor evaluations, we just stop looking based on the notion that
            # our priority list is well put together - we could use the allowance
            # to trim out a portion of moves to check
            
            #len(list_of_legal_moves)
    
            # for every piece that has legal moves
            for i in range(len(list_of_legal_moves)):
                
                start_sq = list_of_legal_moves[i][0]
                dest_sq = list_of_legal_moves[i][1]
                move_mod = list_of_legal_moves[i][2]
                
                # # find if we have already checked this move
                # if [start_sq,dest_sq,move_mod] in existing_moves:
                #     continue
                
                # now in terms of the data array indexes
                start_ind = (start_sq - 21) - 2*((start_sq//10) - 2)
                dest_ind = (dest_sq - 21) - 2*((dest_sq//10) - 2)
                
                # now determine the evaluation of this move
                
                # first, find the evaluation of every piece in view of the start square
                # and the destination square
                
                # find the view of the destination square
                (attack_list,attack_me,
                 defend_me,piece_view) = bl.piece_attack_defend(old_board,dest_sq,6,1)
                
                # the total number of pieces affected is the view of the start sq
                # plus the view of the end square
                piece_view += data_array[start_ind][4]
                piece_view.append(start_sq)
                piece_view.append(dest_sq)
                
                # if it is a capture en passant, make sure we include the captured square
                if move_mod == 3:
                    if white_to_play:
                        piece_view.append(dest_sq-10)
                    else:
                        piece_view.append(dest_sq+10)
                
                # eliminate dupliates in piece_view
                total_view = list(set(piece_view))
                
                # get a copy of the board
                new_board = [x[:] for x in old_board]
    
                # make the move on the board copy
                bl.make_move(new_board,[start_sq,dest_sq],move_mod)
                
                # now the move is made, the person to play changes
                next_to_play = not white_to_play
                
                # # THE WHOLE POINT IS TO REMOVE THIS! DONT FORGET
                # # evaluate the board
                # proper_eval = bl.eval_board_2(new_board,next_to_play)
            
                # make a copy of the original board evaluation
                new_eval = old_eval
                
                # find the new phase_value
                phase,bonus,phase_value = bl.determine_phase(new_board,next_to_play)
                
                # if the phase is transitioning, we need to assess the whole board
                if phase != old_board[100][3]:
                    
                    # testing: don't bother with this under the assumption that
                    #          it will even out (see and False above)
                    
                    new_eval = bl.eval_board_2(new_board,next_to_play)
                    
                else:    # we can assess only pieces that are affected by the move
                
                    # update the board evaluation based on the phase value change
                    new_eval += phase_value - data_array[64][0]
                    
                    # now, loop through the total view and find the evaluation before
                    # and after to see how this move changes things
                    for view in total_view:
                        
                        # what was the old piece evaluation
                        view_ind = (view - 21) - 2*((view//10) - 2)
                        old_value = data_array[view_ind][0]
                        
                        piece_type = new_board[view][1]
                        piece_colour = new_board[view][2]
                        
                        # # FOR TESTING ----------------------------------------
                        # if self.mode == "testing":
                        #     # is the move legal
                        #     (move_is_legal,move_type) = bl.is_legal(piece_type,
                        #                                 piece_colour,
                        #                                 old_board[dest_sq],
                        #                                 move_mod)
                        #     find_check = bl.find_check(new_board,piece_colour,True)
                            
                        #     if not move_is_legal:
                        #         print("This move is not legal!")
                        #         print("The board before the move:")
                        #         bl.print_board(old_board,False)
                        #         print("The board after the move:")
                        #         bl.print_board(new_board,False)
                        #         print("The move is: ", start_sq, ", ",
                        #               dest_sq, ", ", move_mod, ".")
                                
                        #     if find_check:
                        #         print("This move leads to check and is not legal!")
                        #         print("The board before the move:")
                        #         bl.print_board(old_board,False)
                        #         print("The board after the move:")
                        #         bl.print_board(new_board,False)
                        #         print("The move is: ", start_sq, ", ",
                        #               dest_sq, ", ", move_mod, ".")
                        # # ----------------------------------------------------
                        
                        if piece_type == 0:
                            
                            new_value = 0.0
                        
                        else:
                        
                            # now lets find the new piece evaluation
                            (attack_list,attack_me,
                            defend_me,piece_view) = bl.piece_attack_defend(new_board,view,
                                                                        piece_type,
                                                                        piece_colour)
                                                                           
                            # FOR TESTING piece_attack_defend ----------------
                            # convert to cpp
                            if self.mode == "testing":
                                cpp_board = cpc.board_to_cpp(new_board)
                                if piece_colour == 2:
                                    cpp_colour = -1
                                elif piece_colour == 1:
                                    cpp_colour = 1
                                pad_struct = bf.piece_attack_defend(cpp_board, view,
                                                                    piece_type,
                                                                    cpp_colour)
                                (cpp_attack_list,cpp_attack_me,
                                cpp_defend_me,cpp_piece_view) = cpc.cpp_pad_to_py(pad_struct)
                                
                                cpp_lists = [cpp_attack_list,cpp_attack_me,
                                             cpp_defend_me,cpp_piece_view]
                                py_lists = [attack_list,attack_me,
                                            defend_me,piece_view]
                                
                                printed = False
                                
                                # loop through them all except piece_view (for now)
                                # since list compare expects nested lists [[a],[b],...]
                                for i in range(3):
                                    is_correct = self.list_compare(py_lists[i],
                                                                   cpp_lists[i])
                                    if not is_correct:
                                        if i == 0: print("Error with attack list")
                                        elif i == 1: print("Error with attack me")
                                        elif i == 2: print("Error with defend me")
                                        elif i == 3: print("Error with piece view")
                                        printed = True
                                        
                                if printed:
                                    print('List one is python, two is cpp')
                                    print('The square in question is ',view)
                                    print("The board for the above errors is ")
                                    bl.print_board(new_board, False)
                            
                            # ------------------------------------------------
                            
                            (new_value,mate) = bl.eval_piece(new_board,view,next_to_play,bonus,
                                                      piece_type,piece_colour,attack_list,
                                                      attack_me,defend_me)
                            # FOR TESTING checkmate fcn ----------------------
                            if self.mode == "testing2":
                            
                                old_mate = False
                                # check if its mate using the old way
                                available_moves = bl.total_legal_moves(new_board,piece_colour)
                                if available_moves == []:
                                    old_mate = True
                    
                                if mate or old_mate:
                                    if mate and old_mate:
                                        #print('The old and new mate functions agree with each other!')
                                        pass
                                    else:
                                        print('The old and new checkmate functions disagree')
                                        bl.print_board(new_board,False)
                                        print('The old function says',old_mate)
                                        print('The new function says',mate)
                                        # the old function is always right
                                        mate = old_mate
                            #-----------------------------------------------
                            
                            # if eval_piece finds a checkmate
                            if mate:
                                mating_move = True
                                if next_to_play:
                                    new_eval = -100.1
                                    break
                                else:
                                    new_eval = 100.1
                                    break
                                    
                        # update the total evaluation
                        new_eval += new_value - old_value
                    
                # now we have computed the new evaluation of the board
                
                # # for TESTING
                # if abs(new_eval - proper_eval) > 1e-13 and abs(new_eval) < 80:
                #     print('Fuck off, difference was',abs(new_eval-proper_eval))
                
                # WE ARE SORTING CONSIDERATIONS - FASTER AT THE END??
                
                # check if this move is better than any previous moves
                for j in range(len(considerations)):
                    
                    # if its better
                    if new_eval*sign > sign*considerations[j][2]:
                        
                        new_elem = [start_sq,dest_sq,new_eval,move_mod]
                        
                        # save the details of this new move
                        considerations.insert(j,new_elem)
                        # considerations.pop()
                        board_list.insert(j,new_board)
                        # board_list.pop()
                        
                        break
                    
            # we have filled up considerations with all moves, and its sorted
            
            # cut off the last element which is [0,0,-1000,0]
            considerations.pop()
            
            # # now save this data in case we re-evaluate this position!
            # self.all_board_moves[list_num][data_id[1]] = considerations
                    
        # the best moves in this position are now in considerations
        
        # now save this data in case we re-evaluate this position!
        self.all_board_moves[list_num][data_id[1]] = considerations
        
        # considerations is sorted to give the best moves first
        
        # go through the existing moves and new moves to get ids for the best ones
        
        new_id_list = []
        
        # if we have a checkmate
        if mating_move:
            # we only care about one move
            num = 1
        
        #print(considerations)
        
        for u in range(num):
            
            # if we have already created this board
            if (#checked_already and 
                [considerations[u][0],considerations[u][1],considerations[u][3]]
                in self.move_array_list[list_num][data_id[1]]):
                
                # instead lets just look up the hash of the child board
                hash_id = hash(str(board_list[u][:100]))
                
                nan,new_id = self.check_hash_id(data_id,hash_id)
                
                # lets make this node active
                # parent = self.parent_id_list[new_id[0]-list_num][new_id[1]]
                
                for parent in self.parent_id_list[new_id[0]-self.offset][new_id[1]]:
                    
                    # if len(self.parent_id_list[new_id[0]-self.offset][new_id[1]]) > 1:
                    #     test = 4+4
                    #     tst = 4*3
                    
                    # does this parent lead to the root position?
                    if (len(self.parent_id_list[new_id[0]-self.offset][new_id[1]])\
                        == 1 or self.is_root_child(parent)):
                        
                        # if it does, activate this parent
                        self.active_node_array[parent[0]-self.offset][parent[1]]\
                            [parent[2]-2] = True
                        
                        # # now break, assuming only one parent can lead to root
                        # break
                
            else:    # we need to create a new board
            
                # create a new data structure entry for it
                new_id = self.new_create(data_id,considerations[u][:2],
                                      considerations[u][3],board_list[u],
                                      considerations[u][2])
            
            # new_id = self.new_create(data_id,considerations[u][:2],
            #                          considerations[u][3],
            #                          considerations[u][2])
            
            # add the evaluation to the new_id
            new_id.append(considerations[u][2])
            
            # save the location of the new data
            new_id_list.append(new_id)
            
        # if mating_move:
        #     self.active_node_array[new_id[0]-self.offset][new_id[1]].append(True)
        #     self.reactivate_list.append([new_id[0],new_id[1],2])
        
        if mating_move:
            
            # # now save this board evaluation
            # self.eval_array_list[new_id[0]-self.offset][data_id[1]][0] = outcome_eval
            
            # next update the board evaluation in the parent
            for parent in self.parent_id_list[new_id[0]-self.offset][new_id[1]]:
                
                # # overwrite parent evalution
                # self.eval_array_list[parent[0]-self.offset][parent[1]]\
                #     [parent[2]-1] = outcome_eval
            
                # finally, we want this node to remain active
                self.reactivate_list.append(parent)
                
                # this doesn't work since we don't have a three element id
                # we don't have the child id
                # self.reactivate_list.append(data_id)
                
                # new_ids = [[data_id[0], data_id[1], outcome_eval]]
        
            # return True,new_ids
            
        
        return new_id_list,mating_move
    
    def calculate_num_boards(self,depth):
        '''This method calculates how many total boards will be searched with
        the current search parameters'''
        
        total = 0
        
        for d in range(depth):
            _,max_branches = self.set_branch_limits(0,d,True)
            total += max_branches
            
        return total
    
    def calculate_timings(self,depth,width,time_limit):
        '''This method calculates how long the engine has to compute, and sets
        the search parameters so that the search will not exceed this time'''
        
        # how many boards do we have time to check
        num_boards = (time_limit / (self.avg_board_ms/1000.0))//1
        
        # use a rough search to find out what speed factor we need
        speed_factor = 0.5 # starting value
        sf_m2 = 0.0 # speed factor two loops ago (minus 2)
        sf_m1 = 0.0 # speed factor last loop (minus 1)
        
        # limits on what the speed factor can be
        low_limit = 0.4
        high_limit = 0.8
        
        # booleans to help detect infinite loop (drop then up and repeat etc)
        dropped = False
        upped = False
        
        # this is a super hacky linear search but should work
        while sf_m2 != speed_factor:
            
            self.speed_factor = speed_factor
            total_boards = self.calculate_num_boards(depth)
            # print("Total boards was",total_boards)
            
            error = num_boards - total_boards
            # print("The error was", error)
            
            sf_m2 = sf_m1
            sf_m1 = speed_factor
            
            if error > 0:
                speed_factor += 0.01
            else:
                speed_factor -= 0.01
            # print("Speed factor has been updated to",speed_factor)
                
            # check if we exceed our limits
            if speed_factor < low_limit:
                dropped = True
                if upped:
                    print("Not ideal but lets avoid an infinite loop!")
                    break
                depth -= 1
                speed_factor = 0.5
                # print("Dropping depth to ", depth)
            elif speed_factor > high_limit:
                upped = True
                if dropped:
                    print("Not ideal but lets avoid an infinite loop!")
                    break
                depth += 1
                speed_factor = 0.5
                # print("Increasing depth to ", depth)
            
        # roundup
        print("The final depth was ", depth)
        print("The final speed factor was ", speed_factor)
        print("The final number of boards searched is ", total_boards)
        
        return depth
                
    def depth_search(self,base_id,white_to_play,depth,width,allowance,time_lim=[],
                     comb=False):
        '''This function performs a depth search with a depth (moves ahead) and
        width (number of response moves to each move) based on the base_data_id
        that points to the board state to be evaluated, and who plays next'''
        
        # start the clock
        t0 = time.process_time()
        
        # save the input parameters
        self.width = width
        self.depth = depth
        self.allowance = allowance
        self.root_node = base_id
        white_comp = white_to_play      # is the computer white or black?
        
        # clean up the search tree
        if self.delete_old:
            # delete outdated lists from the data structure
            for i in range(base_id[0] - self.offset):
                self.delete()
            # update the offset
            self.offset = base_id[0]
            list_num = base_id[0] - self.offset
        else:
            self.offset = 0
            list_num = base_id[0]
        # deactivate any active nodes and reset counters
        self.deactivate_nodes()
        
        # are we threading?
        if self.threading:
            # import the global flag which signals when to stop
            import global_variables # we care about STOP_THREADS
        
        # check if we are running on a time limit
        if time_lim != []:
            # if so, adjust search parameters
            print("The average board ms is ", self.avg_board_ms)
            depth = self.calculate_timings(depth, width, time_lim)
            print("The depth we will search with is", depth)
        else:
            self.speed_factor = 0.5
        
        # board = self.board_array_list[list_num][base_id[1]]
        
        # # determine what phase of play we are in
        # phase = 1
        
        # # PHASE UPDATE CURRENTLY RUINS HASH SYSTEM: FINAL ITEM IN BOARD?
        # # if both players have castled, or given up rights, phase 2
        # if (board[5][3] == 3 or
        #     not (board[0][3] or board[1][3] or board[2][3] or board[3][3])):
        #     phase = 2
            
        #     # if enough pieces have left the board, phase 3
        #     if np.sum(np.array(board)[:,1]) < 32:
        #         phase = 3
                
        # print('THE PHASE IS ',phase)
                
        # # set the phase
        # self.board_array_list[list_num][base_id[1]][100][3] = phase
        
        
        # best num is curently not used in best_moves
        best_num = width
        
        # look at our starting board, and generate a set of promising moves
        first_id_set,mate = self.best_moves_plus_2(base_id,white_to_play,best_num)
        
        # if this board is already in checkmate
        if first_id_set == True:
            print('Depth search is in checkmate before it started')
            return True,[]
        
        # else the board contains an immediate checkmate
        elif mate == True:
            # no need to search any further
            depth = 0
            # activate all the nodes!
            self.deactivate_nodes(set_to=True)

        # initialise variables
        d = 0
        # id_list = [base_id]
        #print('The length of the first id_set is ',len(id_set))
        id_list = first_id_set[:]
        best_num = width #* 2     # only for the first loop
        break_early = 0
        times_up = False
        cycle_time = 20e-3
        depth_iterations = 0
        active_nodes = best_num

        test_counter = 1
        
        # whilst the target depth has not been met
        while d < depth:
            
            d += 1    # increment the depth counter
            
            # what is the time before we begin searching at this depth
            t_cycle_0 = time.process_time()
            
            # print out our progress, preparing for an updating %
            print('\r       searching for moves at depth',d,end='\r')
            #print('The depth is ',d)

            # update and reset
            next_id_list = []
            id_nums = []
            
            # alternate who is making moves
            white_to_play = not white_to_play
            
            counter = 0
            test_counter_2 = 0

            
            if self.use_width_decay == True:
                width = self.width_decay(d)
                print('Width =',width)
                
            # for every branch left at this new depth of the tree
            for id_item in id_list:
                
                # are we threading
                if self.threading:
                    if global_variables.STOP_THREADS:
                        return
                
                # print out out progress as a percentage
                test_counter += 1
                counter += 1
                print('\r{0:.3}'.format(round((counter*100)/(len(id_list)),2)),
                      '%',end='\r')
                
                # generate the most promising moves from this position
                new_ids,mate = self.best_moves_plus_2(id_item[:2],white_to_play,width)
                
                # if this board is already in checkmate or stalemate
                if new_ids == True:
                    # print('This line leads to checkmate or stalemate, found in depth search')
                    new_ids = mate
                
                # if this board contains a checkmating move
                elif mate == True:
                    pass
                
                # later on encorporate this into best_moves_plus_2
                len_ids = len(new_ids)
                id_nums.append(len_ids)
                    
                # we have finished cascading and trimming, now save whats left
                next_id_list += new_ids
            
            # deactive the sums of the top level ONLY
            # the assumption is that every other level is done during global cascade
            # but the top level is not done because it is updated via parents only
            # is that a correct assumption?? i should probably check...
            for i in range(len(self.num_nodes_array[list_num][base_id[1]][1:])):
                self.num_nodes_array[list_num][base_id[1]][i+1] = 0

            # now update all the upstream evaluations
            self.global_cascade(id_list,white_to_play,d)
            
            # we get errors when the next_id_list becomes empty, lets try to catch them
            if len(id_list) == 0:
                print('The id list was empty!!! Ending search early')
                break
            
            # TESTING CODE DELETE LATER
            total = 0
            # total up the number of active branches using num_nodes_array
            for num in self.num_nodes_array[list_num][base_id[1]][1:]:
                total += num
            self.num_nodes_array[list_num][base_id[1]][0] = total
            
            # REMOVE THIS LATER SINCE IT ALREADY OCCURS AT START OF DEPTH LOOP
            # remove any duplicates from the id_list
            test_list = self.remove_duplicates(next_id_list)
            
            print('The top level is',self.num_nodes_array[list_num][base_id[1]])
            # print('The length of new_ids is', len(next_id_list))
            print('The length of new_ids is', len(test_list),'without duplicates, and with: ',len(next_id_list))
            print('The total active nodes is', total)
            
            # now do the pruning operation
            
            # if we are on the final depth, no need to prune
            if d == depth:
                break

            # method = 'prune'
            # # testing out the branch number limits
            # next_id_list = self.cut_branches(next_id_list,id_nums,white_to_play,
            #                                  white_comp,d,method)
            
            next_id_list = self.global_branch_limits(next_id_list,white_to_play,white_comp,d)
            #next_id_list = self.global_prune(next_id_list,id_nums,white_to_play,d)
            
            print('Following global branch limits, the length of new_ids is',len(next_id_list))
            
            # # remove branches that don't have a good enough evaluation, as well as duplicates
            # next_id_list = self.global_prune(next_id_list,id_nums,white_to_play,d)
            
            # determine the number of active nodes
            active_nodes = 0
            
            # loop through each base node to see if it's active
            for a in range(len(self.active_node_array[base_id[0]-self.offset]\
                               [base_id[1]])):
                # increment the counter for each active node
                if self.active_node_array[base_id[0]-self.offset][base_id[1]][a] == True:
                    active_nodes += 1
            
            # determine if we need to keep searching, based on active nodes
            
            print('There are',active_nodes,'active nodes')
                
            # if we are down to our last branch, take note
            if active_nodes <= 1:
                break_early += 1
                
            if break_early == 2:
                break
                
            # if active_nodes == 0:
            #     d -= 1
            #     self.allowance *= 1.5
            #     next_id_list = id_list[:]
            #     print('We had no active nodes! This shouldnt happen')
            #     continue
                
            # # if we have had two loops down to the last branch, end the search
            # if break_early == 2 or (comb and depth_iterations == 0 
            #                         and d == comb_depth):
            #     # if we have already started again, quit
            #     if depth_iterations >= 1 + comb:
            #         break
            #     else:    # start the depth search from the beginning again
                
            #         # reset key search parameters back to their initial values
            #         break_early = 0
            #         d = 0
            #         white_to_play = white_comp
                    
            #         # keep a record of the restart
            #         depth_iterations += 1
                    
            #         # adjust the search parameters (perhaps based on depth iterations?)
            #         self.allowance = allowance
            #         best_num = width
                    
            #         # find the most promising moves to search through next time
            #         # id_list = first_id_set[:]
            #         id_list = self.moves_to_search(base_id,white_comp,best_num)
                    
            #         print('WE ARE RESTARTING THE DEPTH SEARCH')
            #         continue
                
            # now all the loops at this depth are done
            id_list = next_id_list[:]
            
            # if it is checkmate then no more searching to be done
            if next_id_list == []:
                print('next_id_list in depth search is empty, checkmate perhaps?')
                break
            
            # DELETE LATER: lets find out the current evaluations
            final_move = self.print_engine_considerations(base_id,white_comp,
                                                          active_nodes)
            
            print('\r Done  ')
            
            # if we have run out of time, break
            if times_up:
                break
            else:    # find out how long each cycle is taking
                t_cycle_1 = time.process_time()
                num_cycles = counter
                total_cycle_time = t_cycle_1 - t_cycle_0
                
                # recalculate the cycle time
                cycle_time = total_cycle_time/num_cycles

            
        # now we have finished depth searching, it is time to find the best move
        
        # final out the best moves now, this function is inefficient but provides
        # more information, can be replaced with the below code: however, check
        # that the max/min is correct!
        
        # self.opening_book = True
        
        if base_id[0] < 7 and self.opening_book:
            random_choice = True
            num_print = best_num
        else:
            random_choice = False
            num_print = 3
            
        if num_print > active_nodes:
            num_print = active_nodes
        
        # print the engines thoughts
        print('All done, the top three moves were:')
        final_move,final_id,final_eval = self.print_engine_considerations(base_id,white_comp,
                                                num_print,rand=random_choice)
                                                            
        # what does the engine think is the best reply to its move?
        print('The engine considers the best responses as:')
        self.print_engine_considerations(final_id,not white_comp,3)
        
        # FOR TESTING, print out useful information about the search
        print('The number of boards scanned was',test_counter)
        
        # stop the clock
        t1 = time.process_time()
        
        # save how long each board took for next search
        self.avg_board_ms = ((t1-t0)*1000 / test_counter)
        
        print('The computation time was',t1-t0)
        print('The time per board was {0:.2f} ms'.format(((t1-t0)*1000)/test_counter))
        
        # finally, we are done
        return final_move,final_id,final_eval
                    
    def moves_to_search(self,data_id,white_to_play,best_num):
        '''This function scans the move_array_list to pick out the top
        best_num number of moves and returns their data_ids in a list'''
        
        list_num = data_id[0] - self.offset
        
        # loop through every move child at this data_id
        # minus 1 because eval[0] = id
        loops = len(self.id_array_list[list_num][data_id[1]])-1
        
        # find out how many moves are on offer
        if loops < best_num:
            num = loops
        else:
            num = best_num
            
        # create empty move shell
        if white_to_play:
            new_ids = [[0,0,-1000.0,0] * 1 for p in range(num)]
            sign = 1
        else:
            new_ids = [[0,0,1000.0,0] * 1 for p in range(num)]
            sign = -1
        
        # HASH VERSION
                    
        # loop through every move
        for i in range(loops):
            
            # what is the evaluation of this move
            beta = self.eval_array_list[list_num][data_id[1]][i+1]
            
            # what is the hash_id of this move
            hash_id = self.id_array_list[list_num][data_id[1]][i+1]
            
            move_mod = self.move_array_list[list_num][data_id[1]][i+2][2]
            
            # put it in the list if its better than a previous move
            for j in range(num):
                
                # if this move has a more favourable evaluation
                if beta*sign > sign*new_ids[j][2]:
                    
                    # check what the child id is using the hash
                    exists,new_id = self.check_hash_id(data_id,hash_id)
                    
                    if exists:
                        # i+1 is the id of the move in the original list
                        new_elem = [new_id[0],new_id[1],beta,i+1,move_mod]
                    else:
                        print('No matches found for this move in moves_to_search')
                        new_elem = [0,0,beta,0]
                    
                    # # now find the index of the move child
                    # for k in range(len(self.id_array_list[data_id[0]+1])):
                        
                    #     id_checked_out = False
                        
                    #     # if we find a match, break with this k
                    #     if hash_id == self.id_array_list[data_id[0]+1][k][0]:
                    #         id_checked_out = True
                    #         new_elem = [data_id[0]+1,k,beta]
                    #         break
                    
                    # # if the search failed (it never should!)
                    # if id_checked_out == False:
                    #     print('No matches found for this move in moves_to_search')
                    #     new_elem = [0,0,beta]
                        
                    # save the new element
                    new_ids.insert(j,new_elem)
                    new_ids.pop()
                    break
                
        return new_ids
    
    @staticmethod
    def letters_to_moves(move_order):
        '''This method converts a move order in a2a4 letter format into a
        numerical format which is suitable for the computer'''
        
        num_order = []
        
        for i in range(len(move_order)):
            
            board_str = move_order[i]
            
            # figure out what the move string is
            new_str = []

            if   board_str[1] == '1':  new_str = '2'   
            elif board_str[1] == '2':  new_str = '3'
            elif board_str[1] == '3':  new_str = '4'
            elif board_str[1] == '4':  new_str = '5'
            elif board_str[1] == '5':  new_str = '6'
            elif board_str[1] == '6':  new_str = '7'
            elif board_str[1] == '7':  new_str = '8'
            elif board_str[1] == '8':  new_str = '9'
            
            if   board_str[0] == 'h':  new_str += '1'
            elif board_str[0] == 'g':  new_str += '2'
            elif board_str[0] == 'f':  new_str += '3'
            elif board_str[0] == 'e':  new_str += '4'
            elif board_str[0] == 'd':  new_str += '5'
            elif board_str[0] == 'c':  new_str += '6'
            elif board_str[0] == 'b':  new_str += '7'
            elif board_str[0] == 'a':  new_str += '8'
            
            new_str2 = []

            if   board_str[3] == '1':  new_str2 = '2'   
            elif board_str[3] == '2':  new_str2 = '3'
            elif board_str[3] == '3':  new_str2 = '4'
            elif board_str[3] == '4':  new_str2 = '5'
            elif board_str[3] == '5':  new_str2 = '6'
            elif board_str[3] == '6':  new_str2 = '7'
            elif board_str[3] == '7':  new_str2 = '8'
            elif board_str[3] == '8':  new_str2 = '9'
            
            if   board_str[2] == 'h':  new_str2 += '1'
            elif board_str[2] == 'g':  new_str2 += '2'
            elif board_str[2] == 'f':  new_str2 += '3'
            elif board_str[2] == 'e':  new_str2 += '4'
            elif board_str[2] == 'd':  new_str2 += '5'
            elif board_str[2] == 'c':  new_str2 += '6'
            elif board_str[2] == 'b':  new_str2 += '7'
            elif board_str[2] == 'a':  new_str2 += '8'
            
            # so the move string is
            start_sq = int(new_str)
            dest_sq = int(new_str2)
            
            num_order.append([start_sq,dest_sq])
            
        return num_order
    
    @staticmethod
    def ind_to_letter(board_index):
        '''This method takes a board index [start_sq,dest_sq] and converts it
        to standard letter format eg e2e4'''
        
        start_sq = board_index[0]
        dest_sq = board_index[1]
    
        # figure out what the move string is
        new_str = []
        board_str = str(start_sq)
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
    
    def print_engine_considerations(self,base_id,white_comp,best_num,
                                    top_moves=[],rand=False,print_moves=True):
        '''This method prints the moves being considered by the engine at the
        specified base_ind which points to a board state in board_array_list. 
        It also returns the current best move as final_move'''
        
        list_num = base_id[0] - self.offset
        
        if base_id == None:
            return
        
        # if no top moves specified, check every move in the list
        if top_moves == []:
            # minus 2: one cause len([1])=1 and one cause eval[0] = id
            loops = list(range(len(self.id_array_list[list_num][base_id[1]])-1))
        else:    # only check the moves specified by top_moves
            loops = top_moves
        
        # find out how many moves are on offer
        if len(loops) < best_num:
            num = len(loops)
        else:
            num = best_num
            
        # create empty move shell
        if white_comp:
            # considerations = [[0,0,-1000.0] * 1 for p in range(best_num)]
            sign = 1
        else:
            # considerations = [[0,0,1000.0] * 1 for p in range(best_num)]
            sign = -1
            
        considerations = []
        
        # if there are no children at this id
        if len(self.eval_array_list[list_num][base_id[1]][1:]) == 0:
            print('No moves considered at this id')
            return None
        
        # else:    # manually handle the first element in the below loop
            
        #     i = 0
        #     j = 0
            
        #     beta = self.eval_array_list[list_num][base_id[1]][i+1]
            
        #     start_sq = self.move_array_list[list_num][base_id[1]][i+2][0]
        #     end_sq = self.move_array_list[list_num][base_id[1]][i+2][1]
        #     hash_id = self.id_array_list[list_num][base_id[1]][i+1]
        #     new_elem = [start_sq,end_sq,beta,hash_id]
        
        #     considerations.insert(j,new_elem)
        
        # loop through every move
        # for i in range(loops):
        for i in range(len(self.eval_array_list[list_num][base_id[1]][1:])):
            
            # what is the evaluation of this move
            beta = self.eval_array_list[list_num][base_id[1]][i+1]
            
            # if this node is not active
            if self.active_node_array[list_num][base_id[1]][i] == False:# and
                #abs(self.eval_array_list[list_num][base_id[1]][i+1]) != 100.1):
                continue
            
            # put it in the list if its better than a previous move
            for j in range(len(considerations)+1):
                    
                if (considerations == [] or
                    beta*sign > sign*considerations[j][2]):
                    
                    start_sq = self.move_array_list[list_num][base_id[1]][i+2][0]
                    end_sq = self.move_array_list[list_num][base_id[1]][i+2][1]
                    hash_id = self.id_array_list[list_num][base_id[1]][i+1]
                    new_elem = [start_sq,end_sq,beta,hash_id]
                
                    considerations.insert(j,new_elem)
                    break
                
                elif j == len(considerations)-1:    # on the last element
                
                    start_sq = self.move_array_list[list_num][base_id[1]][i+2][0]
                    end_sq = self.move_array_list[list_num][base_id[1]][i+2][1]
                    hash_id = self.id_array_list[list_num][base_id[1]][i+1]
                    new_elem = [start_sq,end_sq,beta,hash_id]
                
                    considerations.insert(j+1,new_elem)
                    break
                
        # check if this includes any moves
        if considerations == []:
            print('No active nodes at this id')
            return None
        
        final_roundup = considerations[0:num+1]
        
        # if we are chosing the best move randomly
        if rand:
            
            print('I have chosen a move randomly from the list')
            
            import random
            
            # how much worse the move should be before we discount it
            tolerance = 0.3
                
            if white_comp:
                sign = 1
            else:
                sign = -1
            
            # choose a good move randomly
            
            # first, remove any bad moves from consideration
            while True:
                
                if final_roundup[-1][2]*sign < sign*final_roundup[0][2] - tolerance:
                    final_roundup.pop()
                else:
                    break
            
            # # tell the player which moves were chosen between
            # print('I chose randomly between:')
            
            # for ids in id_set:
            #     move = self.move_array_list[ids[0]-self.offset][ids[1]][1][0:2]
            #     move_letters = self.ind_to_letter(move)
            #     print('The move',move_letters,'with evaluation',ids[2])
            
            rand_choice = random.choice(final_roundup)
            
            final_move = rand_choice[:2]
            final_eval = rand_choice[2]
            exists,final_id = self.check_hash_id(base_id,rand_choice[3])
            
        else:    # we do not choose randomly, we pick the best move
        
            final_move = final_roundup[0][:2]
            final_eval = final_roundup[0][2]
            exists,final_id = self.check_hash_id(base_id,final_roundup[0][3])
            
        # final_id = self.check_hash_id(base_id,considerations[0]
        
        # # find the board id of the final move
        # (is_legal,move,move_modifier) = \
        #     bl.interpret_move(self.board_array_list[list_num][base_id[1]],
        #                                                   final_move,
        #                                                   white_comp,False)
        
        # # copy the old board so we can make the move on it
        # new_board = copy.deepcopy(self.board_array_list[list_num][base_id[1]])
        
        # # make the move on the board copy
        # bl.make_move(new_board,move,move_modifier)
        
        # # find the hash ID of the board
        # hash_id = hash(str(new_board[:100]))
        
        # # now lookup the id of the chosen final move
        # exists,final_id = self.check_hash_id(base_id,hash_id)
        
        # now we have all we need, do we print engine considerations?
        
        if print_moves:
        
            # reverse the roundup to show moves worst->best
            final_roundup.reverse()
            
            roundup_string = 'The move {0} has evaluation {1:.4}'
            
            for f in final_roundup:
                
                move_inds = f[:2]
                evaluation = f[2]
                
                move_str = self.ind_to_letter(move_inds)
                
                print(roundup_string.format(move_str,evaluation))
            
        return (final_move,final_id,final_eval)
    
    def checkmate(self,data_id,white_to_play):
        '''Check if the board at data_id is in checkmate'''
        
        board = self.board_array_list[data_id[0]-self.offset][data_id[1]]
        
        mate = bl.checkmate(board,white_to_play)
        
        return mate
    
    def play_timed_game(white_comp,total_time,time_added_per_move=[]):
        '''This method sets the computer up to play a game'''
        return
        










