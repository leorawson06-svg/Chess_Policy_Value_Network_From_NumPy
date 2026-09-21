import numpy as np 
import chess
import pickle
from scipy.special import softmax
import time
import FEN_2_TENSOR as F
import random



with open("move_TENS_dict.pkl", "rb") as f:## this imports the move:tensor dtionary, maps(move.uci,turn) -->tensor
    move_dict = pickle.load(f)

with open("ID_TENS_dict.pkl", "rb") as f:## this imports the move_index:tensor dtionary, maps(moveID,turn) ---> (tensor)
    ID_dict = pickle.load(f)

with open("COORD_MOVE_dict.pkl", "rb") as f:## this imports the move_index:tensor dtionary , maps(coord of 1,turn) ---> move.uci(),index)
    POS_dict = pickle.load(f)

def load_full_model(filename):

    data = np.load(filename, allow_pickle=True)
    for key in data:
        globals()[key] = data[key]
    
    return data

#start_again = False
RESIDUAL_BLOCKS = 12

load_full_model("Example_Model.npz")


def im2col(input, kernel_size=3, padding=1, stride=1):
    
    
    C, H, W = input.shape
    pad = padding
    input_padded = np.pad(input, ((0,0), (pad,pad), (pad,pad)), mode='constant')
    
    out_h = (H + 2*pad - kernel_size)//stride + 1
    out_w = (W + 2*pad - kernel_size)//stride + 1
    
    cols = []
    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            patch = input_padded[:, i:i+kernel_size, j:j+kernel_size]
            cols.append(patch.reshape(-1))  # flatten (C*k*k,)
    
    cols = np.array(cols).T   # shape (C*k*k, out_h*out_w)
    return cols

def conv(tensor,kernels,bias,k,p):
    cols0 = im2col(tensor, kernel_size=k, padding=p)             # (C*9, 64)
    K0_mat = kernels.reshape(kernels.shape[0], -1)                      # (20, C*9)
    z0 = K0_mat @ cols0 + bias[:, None]                          #
    z0 = z0.reshape(kernels.shape[0], 8, 8)
    return z0

def residual_block(batch,filter_1,filter_2,bias_1,bias_2,gam_1,gam_2,bet_1,bet_2):
    n,c,h,w = batch.shape

    ## FIRST PASS
    output_1 = np.zeros((batch.shape))## assuming  we have 8x8 --> 8x8

    for num in range(n):
        output_1[num] = conv(batch[num],filter_1,bias_1,3,1) ## matching the output of eahc convution to a 4D plane in the output tensor, 
    
    ## now output is like a n list of 3_d output tensors, so shape off... batc size,channels(no of kernels),widht,heiight
    

    ## generate batch statistics, 
    mu_1 = np.mean(output_1, axis=(0, 2, 3))  # shape: (20,) ## mean of every channel
    sig2_1 = np.var(output_1, axis=(0, 2, 3)) #shape(20,) ### vraince of each cannnel

    #### BATCH NORMALISATION
    epsilon = 1e-5
    BN_1 = (output_1 - mu_1[None, :, None, None]) / np.sqrt(sig2_1[None, :, None, None] + epsilon)


    ##  appply out = y*z + B

    OUT_1 = gam_1[None, :, None, None] * BN_1 + bet_1[None, :, None, None]


    ## APPPLY RELU 

    rel_out_1= np.maximum(0,OUT_1)
    


    #### SECOND PASS######

    output_2 = np.zeros((batch.shape))## assuming  we have 8x8 --> 8x8

    for num in range(n):
        output_2[num] = conv(rel_out_1[num],filter_2,bias_2,3,1) ## matching the output of eahc convution to a 4D plane in the output tensor, 
    
    ## generate batch statistics, 
    mu_2 = np.mean(output_2, axis=(0, 2, 3))  # shape: (20,) ## mean of every channel
    sig2_2 = np.var(output_2, axis=(0, 2, 3)) #shape(20,) ### vraince of each cannnel

    #### BATCH NORMALISATION
    epsilon = 1e-5
    BN_2 = (output_2 - mu_2[None, :, None, None]) / np.sqrt(sig2_2[None, :, None, None] + epsilon)


    ##  appply out = y*z + B

    OUT_2 = gam_2[None, :, None, None] * BN_2 + bet_2[None, :, None, None]

    ## SKIP CONEECTIOON

    SKIP = OUT_2 + batch


    ## APPPLY RELU 

    rel_out_2= np.maximum(0,SKIP)

   

# x = input
# s1 = output of 1st conv
# b1 = normalised output
# var1,variation of s1
# a1 = activated 1st output
# s2 = raw output of 2nd conv
# b2 =  normalised
# var2 = varaition of output of s2
# u = norla fo 2nd + input 
# y =  actived skip
# OUT_1 = BN_1 wiht gamma and beta included
# OUT_2 = BN_2 with gamm and bet included
# mus are the mean
## mask 1 and 2 are the relu masks that we apply in back prorp, thy consist of 0a dn 1, in elemnt that passed or didnt pass RELU



##sis out put form conv layer and b1 is the norlaised data




    return rel_out_2

def col2im(cols, input_shape, kernel_size=3, padding=1, stride=1):
    C, H, W = input_shape
    pad = padding
    H_p, W_p = H + 2*pad, W + 2*pad
    input_padded = np.zeros((C, H_p, W_p))
    
    out_h = (H + 2*pad - kernel_size)//stride + 1
    out_w = (W + 2*pad - kernel_size)//stride + 1
    
    cols = cols.T.reshape(out_h*out_w, C, kernel_size, kernel_size)
    idx = 0
    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            input_padded[:, i:i+kernel_size, j:j+kernel_size] += cols[idx]
            idx += 1
    return input_padded[:, pad:pad+H, pad:pad+W]

def del2del(batch, x, W, kernel_size=3, padding=1):
    C_out, C_in, k, _ = W.shape
    H, W_in = x.shape[1], x.shape[2]
    
    nums= batch.shape[0]
    #cols = im2col(x, kernel_size, padding)   # ✅ single return
    N = H * W_in


    dw = np.zeros_like(W)
    db = np.zeros((C_out,))
    result = np.zeros((batch.shape[0],C_in,8,8))
   
    for num in range(nums):
        dout = batch[num]
        
        cols = im2col(x[num], kernel_size, padding)   # ✅ single return
        dout_flat = dout.reshape(C_out, -1)   # (C_out, N)
        
        
        dW_ = dout_flat @ cols.T

        
        dw += dW_.reshape(W.shape)

        db += np.sum(dout_flat, axis=1)

        
        W_flat = W.reshape(C_out, -1)
        dcols = W_flat.T @ dout_flat
        
        result[num] = col2im(dcols, x[num].shape, kernel_size, padding)

    
        # after computing dW, dB
    dw /= float(batch.shape[0])
    db /= float(batch.shape[0])
    return db, dw, result
  
def top_3_moves(pred,turn,FEN):

    board = chess.Board(FEN)
    n = 0 
    for move in board.legal_moves:
        n+=1
    
    if n>3:
        n = 3
    

    flat_indices = np.argsort(pred, axis=None)[-n:][::-1]

    # Convert to 3D coordinates
    c = np.unravel_index(flat_indices, pred.shape)
    coords= []
    for j in range(n):
        coords.append([c[0][j].item(),c[1][j].item(),c[2][j].item()])
    

                
    move_scores = []
    
    

    for plane,row,column in coords:
        
        move_scores.append([POS_dict[((plane,row,column),turn)][0],pred[plane][row][column].item()])


    return move_scores

def step_up_conv(input,K_init,B_init):
    n,c,h,w = input.shape
    c_out = K_init.shape[0]
    z_conv = np.zeros((n,c_out,h,w))
    
    for num in range(n):
        z_conv[num] = conv(input[num],K_init,B_init,3,1)
    
    

    RELU_MASK = (z_conv>0)

    out = np.maximum(z_conv,0)

    cache = {"input":input,
             "zconv":z_conv,
             "relu_mask":RELU_MASK}
    
    return out

def POL_FORWARD(batch,KP0,BP0,KP1,BP1,FENS): ##onlyone layer that is batch normalised. 

    ## FENS are the associated FENS, for eac memebr of thr btch, so that our legal move mask can be applied.  

    ## batch contains the batch outputs after the residual blocks

## thhen we convolve to reducte to 30x8x8 each memebr of batch, normalise as norma, then apply, another ocnution to get final 73x8x8 tenosr
## then we apply our legal moves mask, then apply our softmax. 
    n,c,h,w = batch.shape
    a,b,d,e = KP0.shape
    output_1 = np.zeros((n,a,h,w))

    """result = np.sum(batch, axis=1)
    print(result)"""

    for num in range(n): ## process each batch member
        output_1[num] = conv(batch[num],KP0,BP0,3,1)


    ## APPPLY RELU 

    rel_out_1= np.maximum(0,output_1)

    


#### THIS IS OUR FINAL BATCH_NORM reuqired layer,
### now we just step up to our 73x8x8 policy tensor,
    
    
    LARG_NEG = -1e9
    FINAL = np.zeros((n,64,8,8)) ## this will house all the batch reuslts after the soft max over legal moves,

    for num in range(n): ## this loop matches every memebr of batch to its softmxed over legla move tensor, 

        z = conv(rel_out_1[num],KP1,BP1,3,1)## output of final convlution(linear atcivation)

        board = chess.Board(FENS[num]) # create chess object

        legal_moves = np.zeros((64,8,8)) ## ake   moves object
        turn = FENS[num].split(' ')[1]  ### find trn of payer

        for move in board.legal_moves:
            legal_moves += move_dict[move.uci()[:4],turn] ## add move tensor for eveyr legal move
        
        legal_mask = np.where(legal_moves != 0,z,LARG_NEG) ## where tensor =1  replace with conv output if not make it -Larg__neg soo   soft max asigns it small number

        soft = softmax(legal_mask.reshape(-1), axis=0).reshape(64, 8, 8) ## softmax over all legal moves ### could in future add illegal move penalty

        FINAL[num] = soft


        
   



    return FINAL,legal_moves
    ### legal move mask

def VAL_FORWARD_2(batch,KV0,BV0,w0,b0):
    ##VAL_FORWARD will have no BN as this is apparenlty atypical,
    n,c,h,w = batch.shape
    CONV = np.zeros((n,1,h,w)) ## as it is a 1x1 convution we van mapp every trunksample output to a plane of  8x8 boards


    for num in range(n):
        CONV[num]  = conv(batch[num],KV0,BV0,1,0)[0,:,:] ## 1x1 conv soo no addig neccesary
   


    ## now we have a nx8x8 tensor of results,
    
    #so now we have a tensor of nx8x8 continag all the reuslts of ech sample, we can do all of the  
    
    x_flat = CONV.reshape(n,-1) ## reshapes all boards into n 64 evtcors,this way we can porcess baath in one 

    ## sya w0 is (256,64) and b0 = (256,)
    
   
    

    z1 = x_flat@w0.T + b0[None,:]


    ## this way we havve strored all smaples simulatenously thorugh and stord all the pre RELU actvation nicely in a matrix,
    ## this kidn of optimatsation is porbably able to be done thoughot the forward prop,however to do this with the forward tensors woud be above  my paygrade
    ## i wa=ould at that point have to devote lareg time to undertnd or just copy down someting I dont undertand.  

    cache = {'trunk_input':batch,
             'zconv':CONV,
             'actConv':CONV,
             'ANN_input':x_flat,
             'z1':z1,}


    return z1

def full_through(fen):



    global K_init,B_init,KV0,BV0,w0,b0,RESID_gamma,RESID_bet,RESID_B,RESID_K,BP0,BP1,KP0,KP1

    board = np.zeros((1,20,8,8))
    board[0] = F.fen_to_tensor(fen,fen.split(' ')[1])
    FENS = [fen]
    input = step_up_conv(board,K_init,B_init)

    for k in range(RESIDUAL_BLOCKS):
            a = 2*k
            b = 2*k + 1
            
            k_first = RESID_K[a]
            k_second = RESID_K[b]
            bias_first = RESID_B[a]
            bias_second = RESID_B[b]
            gam_1 = RESID_gamma[a]
            gam_2 = RESID_gamma[b]
            bet_1 = RESID_bet[a]
            bet_2 = RESID_bet[b]

            input = residual_block(input,k_first,k_second,bias_first,bias_second,gam_1,gam_2,bet_1,bet_2) ## add the output to the input of next loop 

    res_val= VAL_FORWARD_2(input,KV0,BV0,w0,b0)
    pol_pred,legal_mask = POL_FORWARD(input,KP0,BP0,KP1,BP1,FENS)

    board = chess.Board(fen)
    
    n = board.legal_moves.count()
    
    ## we need to remove all the promtion moves, so we jst treat it as 1 move. 



    ## difficulty in mapping for mulitple ppromtionmoves, so if we hae capture and a strigh pormotionth old logic doesnt work. 

    promotion = 0

    for move in board.legal_moves:
        if move.promotion is not None:
            promotion +=1
        

    if promotion != 0:
        n -= promotion ## remove all the duplicate moves of promtonio to nogt,bishps,queen etc.  
        n += promotion//4## add on th proo=incple promotion  moves


    
    np.where(legal_mask != 0, 1,0)

    pol_pred  = pol_pred * legal_mask

    pol_pred = pol_pred[0]




    flat_indices = np.argsort(pol_pred, axis=None)[-n:][::-1] ## orering the legal moves,  

    
    # Convert to 3D coordinates
    c = np.unravel_index(flat_indices, pol_pred.shape)

 
    coords= []

    for j in range(n):
        coords.append([c[0][j].item(),c[1][j].item(),c[2][j].item()])
    
    turn = fen.split(' ')[1]
                
    move_scores = []
    

    

    #print(fen,res_val)
    for plane,row,column in coords:   
    
   
        
        move_scores.append([POS_dict[((plane,row,column),turn)][0],pol_pred[plane][row][column].item()])
        #print([POS_dict[((plane,row,column),fen.split(' ')[1])][0],pol_pred[plane][row][column].item()])
    #print(res_val)

    res_val = np.atanh(res_val)*10 ## returns centipawn estimate from persepctive of playing player.

    print(fen)
    print(res_val)
    print(move_scores[:3])

    return res_val,move_scores


## Examples ##

#Simple_Captures
#full_through('r3k2r/p4ppp/1p2p3/4PnN1/8/P7/P1Pq1P1P/3R1KR1 b kq - 0 18')
#full_through('2kr3r/ppp3pp/3q1p2/4p3/2P1n3/1P2P3/PB2QPPP/2R2RK1 w - - 0 17')
#full_through('7k/1p6/3p4/3Pp1q1/1P2Q3/1B5p/P1K1r3/6R1 w - - 0 41')

##Simple Mates
##M2##
#full_through('4rk1/p3n1bp/1pB3p1/2p1p1N1/3pQ3/3P4/PPP2qPP/R1B1R2K b - - 1 18')
#full_through('r4rk1/p3n1bp/1pB3p1/2p1p1N1/3pQ3/3P4/PPP3PP/R1B2R1K b - - 0 19')
##M1##
#full_through('8/R6p/2b5/8/7P/Pk4P1/1P3P1K/4r3 b - - 1 47')

## Forks/Tactics
#full_through('1k2r3/pp6/1qp4p/2np2p1/P5P1/1PP4P/1Q6/R1K3R1 b - - 0 1')
#full_through('8/1p4kp/p3p3/6N1/1PPq4/4p1P1/P4RK1/8 w - - 0 43')


