### this code will take a FEN string and convert it to a tensor. 

### I will refer to this format at TENS.

## the TENS of eahc board will 12x8x8  and then have a few additional terms to represent catling and en passant rights. 

### we then have an 8x8 grid for turn to move, castle rights, en passant, so in totla a 18 x 8 x 8 tensor, which we flatten to a 1152 list. 

### so for each 8x8 slice of the TENS will repesent the postions of a given piece. so maye the firts layer, will be used to go down th booard
### starting at thr top left and moveing right unti end of the row and ging down a row and going again. 

## everytime the piece of intrest is found we will make the corresponding entry a 1. 
## then we will want to add a few entries at the bottom to indicte castling rights and enpassant rights. 

#x = '8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -'

#x = '8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -'

import chess
import numpy as np
import time
import random

def sq_coord(sq):
    
    key = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}

    return key[sq[0]],8-int(sq[1]) ## row , col of matrix

def fen_to_tensor(fen,turn):
    board = chess.Board(fen) ### inbuilt function. 
    tensor = np.zeros((12, 8, 8), dtype=np.float32)### we make the tensor 14 deep as we need to indicate en passant rights, white/black to move, catsle rights.
    piece_to_index_WHITE = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }### dictionary if side to move is white

    piece_to_index_BLACK = {
        'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5
    }### dictionary if side to moev is black
  
    
    if turn == 'w':
        ALLY = chess.WHITE
        ENEMY= chess.BLACK
        piece_to_index  = piece_to_index_WHITE
    else:
        ALLY = chess.BLACK
        ENEMY = chess.WHITE
        piece_to_index= piece_to_index_BLACK

    for square in chess.SQUARES:
            piece = board.piece_at(square) 
            if piece: 
                if piece.color == ALLY:
                    row = 7 - (square // 8)
                    col = square % 8 ## modulo 8 
                    index = piece_to_index[piece.symbol()] 
                    tensor[index, row, col] = 1.0
                else :
                    row = 7 - (square // 8)
                    col = square % 8 ## modulo 8 
                    index = piece_to_index[piece.symbol()] 
                    tensor[index, row, col] = -1.0
    
    


    turn_plane = np.full((1, 8, 8), 1 if board.turn == chess.WHITE else 0.0) ## add a 8x8 of 0's to indcate turn to move. 

    ep_plane = np.zeros((1, 8, 8)) # add a 1 if there is an enpassant ove available.
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        ep_plane[0, row, col] = 1.0

    # Add 4 castling channels (white king, white queen, black king, black queen)
    ALLYKS = np.full((1, 8, 8), 1.0 if board.has_kingside_castling_rights(ALLY) else 0.0)
    ALLYQS = np.full((1, 8, 8), 1.0 if board.has_queenside_castling_rights(ALLY) else 0.0)
    ENEMYKS = np.full((1, 8, 8), 1.0 if board.has_kingside_castling_rights(ENEMY) else 0.0)
    ENEMYQS = np.full((1, 8, 8), 1.0 if board.has_queenside_castling_rights(ENEMY) else 0.0)

    #res = attacked_sqs(fen,ALLY)
    #at_ALLY = res[0]
    #at_ENEMY = res[1]

    ##adding the possible moevs of each piece
    # 
 
    ##### THEN WE ADD ALL THE POSSIBLE CAPTURES, AND THEN PIECES OF THE ENEMY WHICH ARE UNDEFENDED.

    capt = []
    enemy_piece = []
    ally_piece = []
    undef_en = np.zeros((1,8,8))
    captures = np.zeros((1,8,8))
    undef_all = np.zeros((1,8,8))

   

    for sq in chess.SQUARES:
            p= board.piece_at(sq)
            if p and p.color == ENEMY:
                enemy_piece.append(sq)
                if board.is_attacked_by(ALLY,sq):
                    capt.append(sq)
            elif p and p.color == ALLY: ## finding peices that are undefened for the ally. 
                ally_piece.append(sq)

        
    for sq in capt:
            r = 7-(sq//8)
            c = sq%8
            captures[0][r][c] = 1


        
    for sq in enemy_piece:
            if not board.is_attacked_by(ENEMY,sq):
                r = 7-(sq//8)
                c = sq%8
                undef_en[0][r][c] = 1
    for sq in ally_piece:
            if not board.is_attacked_by(ALLY,sq):
                r = 7-(sq//8)
                c = sq%8
                undef_all[0][r][c] = 1        

    x = attacked_sqs(fen,turn)



    #tensor = np.concatenate([tensor,turn_plane,ep_plane,ALLYKS,ALLYQS,ENEMYKS,ENEMYQS,captures,undef_en,undef_all], axis=0)
    tensor = np.concatenate([tensor,turn_plane,ep_plane,ALLYKS,ALLYQS,ENEMYKS,ENEMYQS,x[0],x[1]], axis=0)
    ### here our tensor is peces, EP,plane,turnplane,castling, squares attacked vy white and black,then the moves that each piece has for each colcour.  

    # thi sis a costly step as it has too be done every single time however it shouldhelp our netowrk leanr quicker and a provided seriosu reuslts for the value head when seperated previosuly. 

    if ALLY == chess.WHITE:
        pass
    else:
        tensor = np.rot90(tensor, 2, axes=(1, 2))


    return tensor

def attacked_sqs(FEN,turn):### goal of this function is to retun all the sures thhat both sides attack, this way the networ can likly more quickly leanr about  pieces undr attack etc.
    
    ## first half fiinds the squres where ally attacks,  (even if its their own sq)
    
    board = chess.Board(FEN)

    attacked = []
    att = []

    if turn  == chess.WHITE:
        ALLY = chess.WHITE
        ENEMY = chess.BLACK
    else:
        ALLY = chess.BLACK
        ENEMY = chess.WHITE



    for sq in chess.SQUARES:
        if board.is_attacked_by(ALLY, sq): 
            attacked.append(chess.square_name(sq))
            att.append(sq)
    
    attacked_by_ALLY = np.zeros((1,8,8))
    for sq in att:
        r = 7-(sq//8)
        c = sq%8
        attacked_by_ALLY[0][r][c] = 1



    ## sedon hal fdoes the same for the enemy attacks

    attacked = []
    att = []
    for sq in chess.SQUARES:
        if board.is_attacked_by(ENEMY, sq): 
            attacked.append(chess.square_name(sq))
            att.append(sq)
    
    attacked_by_ENEMY = np.zeros((1,8,8))
    for sq in att:
        r = 7-(sq//8)
        c = sq%8
        attacked_by_ENEMY[0][r][c] = 1

    return attacked_by_ALLY,attacked_by_ENEMY


