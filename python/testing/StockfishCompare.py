import cpp_py_conversions_vs as cpc
import board_func as bf
import board_light as bl


class StockfishCompare:

    def __init__(self):

        num = 21

        moves = [[] for i in range(num)]
        board = [[] for j in range(num)]
        engines = [[] for k in range(num)]
        stockfish = [[] for l in range(num)]
        output = [0 for m in range(num)]
        board_eval = [0 for n in range(num)]

        # all from the first game
        moves[0] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6']
        moves[1] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5']
        moves[2] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8']
        moves[3] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6']
        moves[4] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5']
        moves[5] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4']
        moves[6] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4']
        moves[7] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7']
        moves[8] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7']
        moves[9] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4']
        moves[10] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7']
        moves[11] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7']
        moves[12] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6']
        moves[13] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6', 'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3']
        moves[14] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6', 'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3', 'h4h5', 'e7d7', 'c7c8', 'd7c8', 'a3a4', 'c8c7']
        moves[15] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6', 'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3', 'h4h5', 'e7d7', 'c7c8', 'd7c8', 'a3a4', 'c8c7', 'b4b5', 'a6b5', 'a4b5', 'd6d5', 'b5b6', 'c7b6']
        moves[16] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5','c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6',  'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3', 'h4h5', 'e7d7', 'c7c8', 'd7c8', 'a3a4', 'c8c7', 'b4b5', 'a6b5', 'a4b5', 'd6d5', 'b5b6', 'c7b6', 
                     'g2f1', 'd5d4', 'f1g2', 'b6c5', 'g2f1', 'c5c4']
        moves[17] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5','c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6',  'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3', 'h4h5', 'e7d7', 'c7c8', 'd7c8', 'a3a4', 'c8c7', 'b4b5', 'a6b5', 'a4b5', 'd6d5', 'b5b6', 'c7b6', 
                     'g2f1', 'd5d4', 'f1g2', 'b6c5', 'g2f1', 'c5c4',
                     'f1g2', 'd4d3', 'g2f1', 'e5e4', 'f1g2', 'd3d2']
        moves[18] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5','c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6',  'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3', 'h4h5', 'e7d7', 'c7c8', 'd7c8', 'a3a4', 'c8c7', 'b4b5', 'a6b5', 'a4b5', 'd6d5', 'b5b6', 'c7b6', 
                     'g2f1', 'd5d4', 'f1g2', 'b6c5', 'g2f1', 'c5c4',
                     'f1g2', 'd4d3', 'g2f1', 'e5e4', 'f1g2', 'd3d2',
                     'g2f1', 'd2d1', 'f1g2']
        moves[19] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5','c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6',  'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3', 'h4h5', 'e7d7', 'c7c8', 'd7c8', 'a3a4', 'c8c7', 'b4b5', 'a6b5', 'a4b5', 'd6d5', 'b5b6', 'c7b6', 
                     'g2f1', 'd5d4', 'f1g2', 'b6c5', 'g2f1', 'c5c4',
                     'f1g2', 'd4d3', 'g2f1', 'e5e4', 'f1g2', 'd3d2',
                     'g2f1', 'd2d1', 'f1g2', 'd1f3', 'g2h3']
        moves[20] = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'b7b5', 'a4b3', 'g8f6', 'e1g1', 'f8c5','c2c3', 'c8b7', 'd2d4', 'c5b6', 'f1e1', 'e8g8', 'd4d5', 'c6e7', 'b1a3', 'd7d6', 'c1g5', 'h7h6', 'g5h4', 'g7g5', 'h4g3', 'f8e8', 'h2h3', 'b6c5', 'f3h2', 'b5b4', 'a3c4', 'e7g6', 'h2g4', 'f6g4', 'd1g4', 'd8f6', 'b3a4', 'e8e7', 'c3b4', 'c5b4', 'e1e2', 'a8b8', 'a2a3', 'b4c5', 'b2b4', 'c5a7', 'c4a5', 'b8c8', 'a1c1', 'b7a8', 'a5c6', 'e7d7', 'g4f5', 'f6f5', 'e4f5', 'g6f4', 'g3f4', 'g5f4', 'g1h2', 'a8c6', 'a4c6', 'd7d8', 'c6a4', 'd8d7', 'e2c2', 'a7d4', 'c2c7', 'c8c7', 'c1c7', 'd7c7', 'a4c6', 'd4f2', 'g2g3', 'c7c6', 'd5c6', 'f7f6',  'h2g2', 'g8f7', 'c6c7', 'f7e7', 'h3h4', 'f4g3', 'h4h5', 'e7d7', 'c7c8', 'd7c8', 'a3a4', 'c8c7', 'b4b5', 'a6b5', 'a4b5', 'd6d5', 'b5b6', 'c7b6', 
                     'g2f1', 'd5d4', 'f1g2', 'b6c5', 'g2f1', 'c5c4',
                     'f1g2', 'd4d3', 'g2f1', 'e5e4', 'f1g2', 'd3d2',
                     'g2f1', 'd2d1', 'f1g2', 'd1f3', 'g2h3', 'e4e3', 'h3h4']

        stockfish[0] = [0.35, 'b5a4','b5c6','b5c4']
        stockfish[1] = [0.81, 'a2a4','c2c3','d2d3']
        stockfish[2] = [0.25, 'a2a4','c1g5','d4d5']
        stockfish[3] = [0.92, 'g5f6','g5e3','g5h4']
        stockfish[4] = [0.00, 'f3d2','a3c2','f3h2']
        stockfish[5] = [0.36, 'h3g4','d1g4','c3b4']
        stockfish[6] = [1.12, 'e1e3','e1c1','e1f1']
        stockfish[7] = [0.90, 'c4a5','a1b1','a1e1']
        stockfish[8] = [13.2, 'g4d7','c6a7','g4f5']

        # queen trade, pieces begin to leave
        stockfish[9] = [8.54, 'c6a7','f5f6','e2a2']

        # rook hanging
        stockfish[10] = [8.48, 'a4d7','a4c6','c1c6']

        # white now a rook down, no obvious plan for white
        stockfish[11] = [-8.55, 'g2g3','g2g4','a4c6']

        # white has a passed pawn that cannot be stopped (all below are winning big)
        stockfish[12] = [12.4, 'c6c7','b4b5','a3a4']

        # white has an unstoppable passed pawn (but 2nd move is 0.0 and 3rd -2.53)
        # moves are queen promotion, rook promotion, bishop promotion
        stockfish[13] = [20.1, 'c7c8','c7c8','c7c8']

        # white has nothing and it is mate in 14
        stockfish[14] = [-100.0, 'b4b5','a4a5','g2f3']

        # now it is mate in 8
        stockfish[15] = [-100.0, 'g2f1','g2h3','g2h1']

        # now it is mate in 7
        stockfish[16] = [-100.0, 'f1g2','f1e2']

        # now it is mate in 4
        stockfish[17] = [-100.0, 'g2f1', 'g2h3', 'g2h1']

        # now it is mate in 3 (all three moves are mate in 3)
        stockfish[18] = [-100.0, 'd1f3','d1h5','d1e3']

        # now it is mate in 2 (all three moves)
        stockfish[19] = [-100.0, 'g3g2', 'e4e3', 'f2e1']

        # now it is mate in 1 (other moves are mate in 3)
        stockfish[20] = [-100.0, 'g3g2', 'e3e2', 'f2g1']

        self.white_to_play = True
        self.num = num
        self.moves = moves
        self.stockfish = stockfish
        self.board = board
        self.engines = engines
        self.output = output
        self.board_eval = board_eval

        self.generate_boards()

        return

    def generate_boards(self):
        """Create the cpp and python boards"""

        self.cpp_boards = [[] for i in range(self.num)]
        self.py_boards = [[] for j in range(self.num)]

        for k in range(self.num):
            self.py_boards[k] = bl.create_board(self.moves[k], True)
            self.cpp_boards[k] = cpc.board_to_cpp(self.py_boards[k])

        return

    def compare(self, start, end):
        """Run the boards from start to end"""

        for i in range(start, end):

            # which boards are we looking at
            py_board = self.py_boards[i]
            cpp_board = self.cpp_boards[i]

            # print both boards for comparison
            print("Board", i, "is:")
            bl.print_board(py_board, True)
            print("The equivalent cpp board is")
            bf.print_board(cpp_board, False)

            # now evaluate the boards
            cpp_eval = bf.eval_board(cpp_board, self.white_to_play)
            print("The cpp evaluation is ", cpp_eval, "( ", cpp_eval/1000, " )")
            py_eval = bl.eval_board_2(py_board, self.white_to_play)
            print("The py evaluation is ", py_eval)
            print("The stockfish evaluation is ", self.stockfish[i][0])

            # now look at piece evaluations in the boards
            print("The python piece evaluations are:")
            cpc.print_py_board_evals(py_board, self.white_to_play)
            print("The cpp piece evaluations are:")
            cpc.print_cpp_board_evals(cpp_board, self.white_to_play)


