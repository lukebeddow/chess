import pygame
import random
import time
from sys import exit
from threading import Thread

import modules.board_module as bf
import modules.tree_module as tf

class ExitException(Exception):
    def __init__(self):
        pass

class ChessWindow:

    class Timer:
        def __init__(self):
            self.t1 = time.perf_counter()
        def click(self):
            self.t2 = time.perf_counter()
            lap_time = self.t2 - self.t1
            self.t1 = time.perf_counter()
            return lap_time
        def since_click(self):
            return time.perf_counter() - self.t1

    class ChessClock:
        def __init__(self, time_per_player, increment=0):
            self.white_time = time_per_player
            self.black_time = time_per_player
            self.white_playing = True
            self.increment = increment
            self.timer = ChessWindow.Timer()

        def click(self):
            if self.white_playing:
                self.white_time -= self.timer.click()
                self.white_time += self.increment
            else:
                self.black_time -= self.timer.click()
                self.black_time += self.increment
            self.white_playing = not self.white_playing
            self.last_click = time.perf_counter()

        def get_time_left(self):
            time_now = time.perf_counter()
            if self.white_playing:
                white_time = self.white_time - self.timer.since_click()
                black_time = self.black_time
            else:
                white_time = self.white_time
                black_time = self.black_time - self.timer.since_click()
            return white_time, black_time

    # ----- functions ----- #
    def __init__(self):

        pygame.init()

        # the size of the game window is defined by the square height
        self.square_h = 75
        self.piece_size = 0.75

        # make the pieces 80% the size of the squares
        self.piece_h = round(self.piece_size * self.square_h)

        # root = r'C://Users//lukeb//OneDrive - University College London//Documents//Chess//128h//'
        root = "/home/luke/chess/pieces/"

        # add in the pieces
        #self.wP = pygame.image.load(r'C:\Users\lukeb\OneDrive - University College London\Documents\Chess\128h\w_pawn_png_128px.png')
        self.wP = pygame.image.load(root + 'w_pawn_png_128px.png')
        self.wN = pygame.image.load(root + 'w_knight_png_128px.png')
        self.wB = pygame.image.load(root + 'w_bishop_png_128px.png')                      
        self.wR = pygame.image.load(root + 'w_rook_png_128px.png')
        self.wQ = pygame.image.load(root + 'w_queen_png_128px.png')                       
        self.wK = pygame.image.load(root + 'w_king_png_128px.png')
        self.bP = pygame.image.load(root + 'b_pawn_png_128px.png')                       
        self.bN = pygame.image.load(root + 'b_knight_png_128px.png')                       
        self.bB = pygame.image.load(root + 'b_bishop_png_128px.png')                       
        self.bR = pygame.image.load(root + 'b_rook_png_128px.png')                       
        self.bQ = pygame.image.load(root + 'b_queen_png_128px.png')                       
        self.bK = pygame.image.load(root + 'b_king_png_128px.png')

        # add the squares
        #self.wS = pygame.image.load(root + 'square brown light_png_128px.png')
        #self.bS = pygame.image.load(root + 'square brown dark_png_128px.png')
        self.wS = pygame.image.load(root + 'square gray light _png_128px.png')
        self.bS = pygame.image.load(root + 'square gray dark _png_128px.png')

        # scale images to the correct size
        self.wP = pygame.transform.scale(self.wP, (self.piece_h, self.piece_h))
        self.wN = pygame.transform.scale(self.wN, (self.piece_h, self.piece_h))
        self.wB = pygame.transform.scale(self.wB, (self.piece_h, self.piece_h))
        self.wR = pygame.transform.scale(self.wR, (self.piece_h, self.piece_h))
        self.wQ = pygame.transform.scale(self.wQ, (self.piece_h, self.piece_h))
        self.wK = pygame.transform.scale(self.wK, (self.piece_h, self.piece_h))
        self.bP = pygame.transform.scale(self.bP, (self.piece_h, self.piece_h))
        self.bN = pygame.transform.scale(self.bN, (self.piece_h, self.piece_h))
        self.bB = pygame.transform.scale(self.bB, (self.piece_h, self.piece_h))
        self.bR = pygame.transform.scale(self.bR, (self.piece_h, self.piece_h))
        self.bQ = pygame.transform.scale(self.bQ, (self.piece_h, self.piece_h))
        self.bK = pygame.transform.scale(self.bK, (self.piece_h, self.piece_h))
        self.wS = pygame.transform.scale(self.wS, (self.square_h, self.square_h))
        self.bS = pygame.transform.scale(self.bS, (self.square_h, self.square_h))

        # set pygame variables
        self.surface_h = 8 * self.square_h
        self.surface_w = 11 * self.square_h
        self.my_font = pygame.font.SysFont('Courier', 20)
        self.main_surface = pygame.display.set_mode((self.surface_w, self.surface_h))
        self.window_box = (8 * self.square_h,
                           0 * self.square_h,
                           3 * self.square_h,
                           8 * self.square_h)
        self.shadow_colours = ((174,101,21),
                               (204,122,51))

        # create a gameboard object to play the game upon
        self.game_board = tf.GameBoard()
        self.white_comp = False

        # threading details
        self.stop_threads = False
        self.timer_thread_running = False
        self.stop_timer_thread = False
        self.engine_thread_running = False
        self.stop_engine_thread = False
        self.monitor_thread_running = False
        self.stop_monitor_thread = False

    def quit_game(self):
        self.stop_threads = True
        if self.timer_thread_running:
            self.timer_thread.join()
        #if self.engine_thread_running:
        #    self.engine_thread.join()
        if self.monitor_thread_running:
            self.monitor_thread.join()
        #self.monitor_thread.join()
        pygame.quit()
        time.sleep(500 / 1000.0)
        raise ExitException

    def square_to_coord(self, square):
        """
        convert a square (eg 'a5') to coordinates x and y
        """

        if square[0] == 'a': i = 0
        elif square[0] == 'b': i = 1
        elif square[0] == 'c': i = 2
        elif square[0] == 'd': i = 3
        elif square[0] == 'e': i = 4
        elif square[0] == 'f': i = 5
        elif square[0] == 'g': i = 6
        elif square[0] == 'h': i = 7
        else:
            raise RuntimeError("bad input to function")

        if square[1] == '1': j = 0
        elif square[1] == '2': j = 1
        elif square[1] == '3': j = 2
        elif square[1] == '4': j = 3
        elif square[1] == '5': j = 4
        elif square[1] == '6': j = 5
        elif square[1] == '7': j = 6
        elif square[1] == '8': j = 7
        else:
            raise RuntimeError("bad input to function")

        # if the human is black
        if self.white_comp:
            x = self.square_h * (7 - i)
            y = self.square_h * j
        else:
            x = self.square_h * i
            y = self.square_h * (7 - j)

        return x, y

    def coord_to_square(self, x, y):
        """
        Finds the square under certain x,y coordinates
        """

        i = x // self.square_h
        j = y // self.square_h

        # if the human is black, swap the squares
        if self.white_comp:
            i = (7 - i)
            j = (7 - j)

        if   i == 0: s1 = 'a'
        elif i == 1: s1 = 'b'
        elif i == 2: s1 = 'c'
        elif i == 3: s1 = 'd'
        elif i == 4: s1 = 'e'
        elif i == 5: s1 = 'f'
        elif i == 6: s1 = 'g'
        elif i == 7: s1 = 'h'
        else: return None
                    
        if   j == 0: s2 = '8'
        elif j == 1: s2 = '7'
        elif j == 2: s2 = '6'
        elif j == 3: s2 = '5'
        elif j == 4: s2 = '4'
        elif j == 5: s2 = '3'
        elif j == 6: s2 = '2'
        elif j == 7: s2 = '1'
        else: return None

        return s1 + s2

    def render_square(self, square, shadow_colour=None, piece_override=None):
        """
        render a square on the board
        """

        # get the coordinates of the square
        x, y = self.square_to_coord(square)
        square_coords = (x, y, self.square_h, self.square_h)

        is_white = (x % 2 + y % 2) % 2

        # put the square on the main surface
        if shadow_colour == None:
            if is_white:
                self.main_surface.blit(self.wS, square_coords)
            else:
                self.main_surface.blit(self.bS, square_coords)
        else:
            self.main_surface.fill(shadow_colour, square_coords)

        # find what piece is on this square
        piece = self.game_board.get_square_piece(square)

        if piece == "empty" and piece_override == None: return
        if piece_override == "empty": return

        # find the coordinates of where the piece will go
        add_on = int((self.square_h * (1 - self.piece_size)) / 2.0)
        piece_coords = (x + add_on, y + add_on, self.square_h, self.square_h)

        # render the piece on the board
        if piece_override != None:
            exec_args = 'self.' + piece_override + ', ' + str(piece_coords)
            exec('self.main_surface.blit(' + exec_args + ')')
        elif piece == "white pawn": self.main_surface.blit(self.wP, piece_coords)
        elif piece == "white knight": self.main_surface.blit(self.wN, piece_coords)
        elif piece == "white bishop": self.main_surface.blit(self.wB, piece_coords)
        elif piece == "white rook": self.main_surface.blit(self.wR, piece_coords)
        elif piece == "white queen": self.main_surface.blit(self.wQ, piece_coords)
        elif piece == "white king": self.main_surface.blit(self.wK, piece_coords)
        elif piece == "black pawn": self.main_surface.blit(self.bP, piece_coords)
        elif piece == "black knight": self.main_surface.blit(self.bN, piece_coords)
        elif piece == "black bishop": self.main_surface.blit(self.bB, piece_coords)
        elif piece == "black rook": self.main_surface.blit(self.bR, piece_coords)
        elif piece == "black queen": self.main_surface.blit(self.bQ, piece_coords)
        elif piece == "black king": self.main_surface.blit(self.bK, piece_coords)
        else:
            print(piece)
            raise RuntimeError("bad input to render_square function")

    def display_moves(self, move_list_letters):
        '''This function prints the text containing the moves made onto the board'''
    
        screen_limit = 20
    
        # where to start printing text from
        top = 595 - screen_limit * 25
    
        # wipe the slate clean
        the_square = (8*75, 0*75, 3*75, 8*75)
        self.main_surface.fill((255,255,255), the_square)
    
        line = 0
    
        # loop through every move
        for i,move in enumerate(move_list_letters):
        
            if len(move_list_letters[i:]) > screen_limit * 2:
                continue
        
            # what turn number is it
            turn = (i // 2) + 1
        
            # is it white or black
            if i % 2 == 0:
            
                line += 1
            
                # it is a white move
            
                if turn < 10:
                    turn_string = ' '+str(turn)
                else:
                    turn_string = str(turn)
            
                row_string = turn_string + '. ' + move
            
                text = self.my_font.render(row_string, True, (0,0,0))
                self.main_surface.blit(text, (17 + (8 * 75), top + (25 * (line - 1))))
            
            else:    # it is a black move
        
                if line == 0:
                    continue
            
                text = self.my_font.render(move, True, (0,0,0))
                self.main_surface.blit(text, (150 + (8 * 75), top + (25 * (line - 1))))
        
        pygame.display.flip()
    
        return

    def create_textbox(self, text_prompts):
        '''This function puts a textbox on the screen, and fills it with the given
        text prompts'''
    
        colours = ((253,222,238),
                   (165,137,193),
                   (249,140,182),
                   (135,107,163),
                   (219,110,152))
    
        # what prompts do we have
        num = len(text_prompts)
    
        # calculate how big the box needs to be
        font_scale = 12
        # space needed to write the whole question
        q_size = font_scale * len(text_prompts[0])
        a_max = 0
        for t in text_prompts[1:]:
            if len(t) > a_max:
                a_max = len(t)
        # space needed to write all the answers
        a_size = font_scale * a_max * (num - 1)
    
        # which takes up more space, question or answer?
        if a_size > q_size:
            textbox_width = a_size * (1.0 + num * 0.1)
        else:
            textbox_width = q_size * (1.0 + num * 0.1)
        # in either case
        textbox_height = 2 * self.square_h
    
        # hence, determine the size of the answer boxes
        ans_width = textbox_width / (num-1.0)
        ans_height = textbox_height / 2.0
    
        # find the centre of the board
        centre = 4 * self.square_h
    
        # find the x0,y0 coordinates of the textbox
        textbox_x0 = centre - textbox_width / 2.0
        textbox_y0 = centre - textbox_height / 2.0
    
        # create the textbox
        textbox = (textbox_x0, textbox_y0, textbox_width, textbox_height)
        self.main_surface.fill(colours[0], textbox)
    
        # data structure to contain the sizes of the answer boxes
        ans_boxes = [ [0,0,0,0] for j in range(num-1)]
    
        # assign the text and make the answer boxes
        for i in range(num):
        
            text = "{0:^}".format(text_prompts[i])
            rtext = self.my_font.render(text, True, (0,0,0))
        
            # if we are putting the question up
            if i == 0:
                # no need to make an answer box
                start_x = centre - font_scale * (len(text_prompts[i]) / 2.0)
                start_y = centre - 0.25 * textbox_height - font_scale
            else:
                # we need to make an answer box
            
                # randomise the colour to something bright so the black text shows
                rand_colour = (random.randint(100,255),
                               random.randint(100,255),
                               random.randint(100,255))
            
                # determine the size of the box
                ans_boxes[i-1][0] = textbox_x0 + (i - 1) * ans_width
                ans_boxes[i-1][1] = textbox_y0 + textbox_height / 2.0
                ans_boxes[i-1][2] = ans_width
                ans_boxes[i-1][3] = ans_height
            
                # create the box
                self.main_surface.fill(rand_colour, ans_boxes[i-1])
            
                # now we need to put the text in the middle of the box
                ans_centre_x = ans_boxes[i-1][0] + ans_width / 2.0
                ans_centre_y = ans_boxes[i-1][1] + ans_height / 2.0 - font_scale
            
                start_x = ans_centre_x - font_scale*(len(text_prompts[i]) / 2.0)
                start_y = ans_centre_y
        
            # put the text on the screen
            self.main_surface.blit(rtext, (start_x, start_y))
        
        # display to the screen
        pygame.display.flip()
    
        # wait for a mouse click to choose
        while True:
        
            time.sleep(10/1000)
        
            # look for an event
            ev = pygame.event.poll()
        
            if ev.type == pygame.QUIT:
                self.quit_game()
        
              # find where the mouseclick is
            if ev.type == pygame.MOUSEBUTTONDOWN:
            
                posn_of_click = ev.dict['pos']
            
                i = posn_of_click[0]
                j = posn_of_click[1]
            
                box_clicked = -1
            
                # loop through all the possible answer boxes
                for box_dim in ans_boxes:
                
                    box_clicked += 1
                
                    if (i > box_dim[0] and i < box_dim[0]+box_dim[2] and
                        j > box_dim[1] and j < box_dim[1]+box_dim[3]):
                    
                        # highlight the clicked box
                        self.main_surface.fill((100,100,100), box_dim)
                        pygame.display.flip()
                        time.sleep(150 / 1000)
                    
                        # return whichever box was clicked
                        return box_clicked

    def render_board(self):
        '''This function renders the board and places it on the screen'''

        for file in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            for rank in ['1', '2', '3', '4', '5', '6', '7', '8']:
                self.render_square(file + rank)

        # put shadows on the board to indicate the previous move
        last_move = self.game_board.get_last_move()
        if last_move != "none":
            self.render_square(last_move[0:2], shadow_colour=self.shadow_colours[0])
            self.render_square(last_move[2:4], shadow_colour=self.shadow_colours[1])
    
        # display the board
        pygame.display.flip()
    
        return

    def render_times(self, human, computer):
        '''This function updates the clock display to keep track of time left'''

        # wipe the timer area
        area_to_wipe = (8 * self.square_h, 0 * self.square_h,
                        3 * self.square_h, 1 * self.square_h)
        self.main_surface.fill((200, 200, 200), area_to_wipe)
    
        # break it down to mins and secs
        human = int(human//1)
        computer = int(computer//1)
    
        h_min = int(human//60)
        h_sec = human - h_min*60
    
        c_min = int(computer//60)
        c_sec = computer - c_min*60
    
        intro_text = "Time (min:sec):"
    
        if h_sec > 9:
            human_text = "Human:    {0}:{1}".format(h_min,h_sec)
        else:
            human_text = "Human:    {0}:0{1}".format(h_min,h_sec)
        if c_sec > 9:
            comp_text =  "Computer: {0}:{1}".format(c_min,c_sec)
        else:
            comp_text =  "Computer: {0}:0{1}".format(c_min,c_sec)
    
        rtext0 = self.my_font.render(intro_text, True, (0,0,0))
        rtext1 = self.my_font.render(human_text, True, (0,0,0))
        rtext2 = self.my_font.render(comp_text, True, (0,0,0))
    
        line = 1
        self.main_surface.blit(rtext0, (15 + (8 * 75), (25 * (line - 1))))
        line = 2
        self.main_surface.blit(rtext1, (15 + (8 * 75), (25 * (line - 1))))
        line = 3
        self.main_surface.blit(rtext2, (15 + (8 * 75), (25 * (line - 1))))
    
        pygame.display.flip()
    
        return

    def display_times(self):
        """
        This function shows the time left for each player
        """

        # get how long each player has left
        white_time, black_time = self.chess_clock.get_time_left()

        # if game is not timed, there is no limit, so times are negative
        if not self.timed_game:
            white_time *= -1
            black_time *= -1

        # which colour is the human
        if self.white_comp:
            self.render_times(black_time, white_time)
        else:
            self.render_times(white_time, black_time)

    def run_timer(self):
        """
        This function is meant to run in a seperate thread, it ticks the timers on screen
        """
        self.timer_thread_running = True
        while True:
            time.sleep(1.0)
            self.display_times()
            if self.stop_threads or self.stop_timer_thread:
                self.timer_thread_running = False
                self.stop_timer_thread = False
                return

    def run_engine(self):
        """
        This function runs the chess engine in a seperate thread
        """
        if self.timed_game:
            target_time = 5
        else:
            target_time = 10 # don't want it to take too long

        self.engine_thread_running = True
        engine_move = self.game_board.get_engine_move(target_time)
        self.engine_thread_running = False
        self.engine_move = engine_move
        return

    def run_monitor(self):
        """
        This function runs the window monitor in a seperate thread
        """
        self.monitor_thread_running = True

        while True:

            time.sleep(100 / 1000.0)

            if self.stop_threads or self.stop_monitor_thread:
                self.monitor_thread_running = False
                self.stop_monitor_thread = False
                return

            ev = pygame.event.poll()

            if ev.type == pygame.QUIT:
                self.quit_game()

            if ev.type == pygame.KEYDOWN:
                key = ev.dict['key']
                    
                if key == ord('u'):
                    pass

                elif key == ord('n'):
                    pass

            if ev.type == pygame.MOUSEBUTTONDOWN:
                pass

    def wait_for_input(self):
        """
        This function waits for a human input into the GUI
        """

        while True:

            ev = pygame.event.poll()

            if ev.type == pygame.QUIT:
                self.quit_game()

            if ev.type == pygame.KEYDOWN:
                key = ev.dict['key']
                    
                if key == ord('u'):
                    self.undo_move()
                     
                elif key == ord('n'):
                    self.new_game()

            if ev.type == pygame.MOUSEBUTTONDOWN:
                posn_of_click = ev.dict['pos']
                return posn_of_click

    def check_game_outcome(self):
        """
        check if the game is over
        """

        outcome = self.game_board.get_outcome()

        if outcome != "continue":
            
            if outcome == "white wins":
                if self.white_comp == True:
                    end_game_text = "The computer wins with white!"
                else:
                    end_game_text = "The human wins with white!"

            elif outcome == "black wins":
                if self.white_comp == True:
                    end_game_text = "The human wins with black!"
                else:
                    end_game_text = "The computer wins with black!"

            elif outcome == "draw":
                end_game_text = "The game is a draw"

            end_game_text += " Choose:"
            prompts = (end_game_text, "Play again", "Quit", "Admire board")
            ans = self.create_textbox(prompts)

            if ans == 0: self.new_game()
            elif ans == 1: self.quit_game()

            # if chosen to admire the board
            elif ans == 2:
                # remove the text box by re-rendering the board
                self.render_board()
                while True:
                    self.wait_for_input()

    def new_game(self, move_list=[]):
        """
        This function resets everything and creates a new game
        """

        # wait to shut down the timer thread, if it is running
        if self.timer_thread_running:
            self.stop_threads = True
            self.timer_thread.join()

        # wipe the window white
        self.main_surface.fill((255, 255, 255), self.window_box)

        # reset the game board
        self.game_board.reset(move_list)
        self.render_board()
            
        # first ask what the time control is
        prompts = ("What time control?", "1|5", "5|5", "10min", "30min","None")
        ans = self.create_textbox(prompts)

        # remove the textbox by re-rendering the board
        self.render_board()

        self.timed_game = True
        if ans == 0:   time_per_player = 1 * 60.0;  increment = 5.0;
        elif ans == 1: time_per_player = 5 * 60.0;  increment = 5.0;
        elif ans == 2: time_per_player = 10 * 60.0; increment = 0;
        elif ans == 3: time_per_player = 30 * 60.0; increment = 0;
        elif ans == 4: 
            self.timed_game = False
            time_per_player = 0.0;     
            increment = 0;
            
        # now find out who goes first
        prompts  = ("What colour will you play as?", "White", "Black")
        ans = self.create_textbox(prompts)

        # remove the textbox by re-rendering the board
        self.render_board()

        if ans == 0:    self.white_comp = False
        elif ans == 1:  self.white_comp = True

        # start a fresh clock
        self.chess_clock = ChessWindow.ChessClock(time_per_player, increment)

        # start the timer thread
        self.timer_thread = Thread(target=self.run_timer)
        self.timer_thread.start()

        # go into the move loop
        self.move_loop()

    def undo_move(self):
        """
        this function undoes a move just made by the human
        """

        # undo two moves on the game board
        self.game_board.undo(2)
        
        # have we wiped the board to a new game state
        if len(self.game_board.get_move_list()) <= 1:
            self.new_game()
        else:
            # return to looping through moves
            self.move_loop()

    def get_human_move(self):
        """
        Get a move input from the human using the GUI
        """

        # initialise
        click_str = ''

        while True:

            posn_of_click = self.wait_for_input()

            # get the square that has been clicked on
            square_clicked = self.coord_to_square(posn_of_click[0], posn_of_click[1])
                    
            # check if this click was out of bounds
            if square_clicked == None:
                print("out of bounds!")
                continue
                    
            # otherwise, this is the selected square
            click_str += square_clicked
                    
            # if one square only has been selected
            if len(click_str) == 2:

                # check if the square contains a piece that can be moved
                sq_colour = self.game_board.get_square_colour(square_clicked)
                if (sq_colour == 0 or 
                    sq_colour != (self.game_board.get_white_to_play() * 2) - 1):
                    click_str = ''
                    continue # if not
                        
                # light up the square selected
                self.render_square(square_clicked, shadow_colour=self.shadow_colours[0])
                        
                #now the surface is ready, tell pygame to display it
                pygame.display.flip()
                        
                # delay so people don't double click
                time.sleep(50/1000)
                        
            # if both square have been selected
            elif len(click_str) == 4:

                # check if the move is a promotion
                try:
                    is_promote = self.game_board.check_promotion(click_str)
                except:
                    click_str = ''
                    continue

                # if it is a promotion, we need to ask the player what to
                if is_promote:
                    # remove the shadows from the last move
                    prev_move = self.game_board.get_last_move()
                    self.render_square(prev_move[0:2])
                    self.render_square(prev_move[2:4])
                    # remove the pawn from the starting square
                    self.render_square(click_str[:2], shadow_colour=self.shadow_colours[0],
                                       piece_override="empty")
                    # render the pawn moving on the board
                    if self.game_board.get_white_to_play():
                        self.render_square(square_clicked, shadow_colour=self.shadow_colours[1], 
                                           piece_override='wP')
                    else:
                        self.render_square(square_clicked, shadow_colour=self.shadow_colours[1], 
                                           piece_override='bP')
                    # display the changes
                    pygame.display.flip()

                    # ask the player what they want to promote to
                    prompts = ["What piece will you promote to?", "Knight",
                                "Bishop", "Rook", "Queen"]
                    ans = self.create_textbox(prompts)
                    if ans == 0: click_str += 'n'
                    elif ans == 1: click_str += 'b'
                    elif ans == 2: click_str += 'r'
                    elif ans == 3: click_str += 'q'
                    else:
                        raise RuntimeError("error from textbox")

                # remove the highlighted square
                self.render_square(click_str[0:2])
                pygame.display.flip()

                # try to make the move on the board
                is_legal = self.game_board.move(click_str)

                # if it was not a legal move, reset and wait for new input
                if is_legal: return
                else:
                    # have we selected another viable piece to move
                    sq_colour = self.game_board.get_square_colour(square_clicked)
                    if (sq_colour == 0 or 
                        sq_colour != (self.game_board.get_white_to_play() * 2) - 1):
                        click_str = ''
                    else:
                        # light up the square selected
                        self.render_square(square_clicked, shadow_colour=self.shadow_colours[0])
                        click_str = square_clicked
                        pygame.display.flip()

    def get_engine_move(self):
        """
        Ask the engine to play the next move
        """

        self.monitor_thread = Thread(target=self.run_monitor)
        self.monitor_thread.start()

        self.engine_thread = Thread(target=self.run_engine)
        self.engine_thread.start()

        #engine_move = self.game_board.get_engine_move_no_GIL()
        #self.game_board.move(engine_move)

        self.engine_thread.join()
        self.game_board.move(self.engine_move)

        self.stop_monitor_thread = True
        self.monitor_thread.join()

    def move_loop(self):
        """
        This function plays the next move of the chess game
        """

        while True:

            ev = pygame.event.poll()    # look for an event

            if ev.type == pygame.QUIT:  # if window close button clicked
                self.quit_game()
        
            # render and display the board
            self.render_board()

            # paste the move text and timing information on screen
            self.display_moves(self.game_board.get_move_list())
            self.display_times()

            # check if the game is over
            self.check_game_outcome()

            # check whos turn it is to play
            if self.game_board.get_white_to_play() == self.white_comp:
                self.get_engine_move()
            else:
                self.get_human_move()

            # stop the clock for this player, start it for the next
            self.chess_clock.click()
        
            # handy for testing to be able to recreate boards
            print(self.game_board.get_move_list())

    def start(self, moves=None):
        """
        Run the new game in a try/except loop
        """

        if moves == None:
            moves = []

        try:
            self.new_game(moves)
        except ExitException:
            print("Game window has been shut down")

if __name__ == "__main__":

    window = ChessWindow()

    try:
        window.new_game()
    except ExitException:
        print("Game window has been shut down")