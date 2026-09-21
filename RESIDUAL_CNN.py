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
    print(f"model loaded from {filename}")
    return data

start_again = False ## if we wnat to begin training from scratch 
RESIDUAL_BLOCKS = 12
if start_again:
    #res = POL_back(tens_pred,tens_target,input_to_final,KP1,b1_pol,s1_pol,var_pol,1e-6,trunk_out,KP0,gam_1,bias0)
    #res = value_back(v_true,v_pred,w2_val,z2,hidden,z1,w1_val,ANN_input,trunk_out,filter)
    #res= residual_back(x,w1,s1,var1,bn1,a1,w2,s2,var2,u,gy,gamma1,gamma2,1e-9)
    ########
    #finalising network architecure
    ########

    ## input  of  20x8x8
    ## convolve too 256x8x8
    ##  12 residual blocks
    ## then split  into two ead
    ## vvalue: 1x1 convoolution  t 8x8 board, then pas horugh FF of 64->256-> 1
    ## policy: 3x3 convolution to 30x8x8()batchnomalistaion
    ## convolution to 73x8x8 (linear then legal mask and  soft max)

    
    small = 0.01
    ##POLICY WEIGHTS
    KP0 = np.random.randn(30,256,3,3) 
    BP0 = np.zeros((30,))
    KP1 = np.random.randn(64,30,3,3)## final policy output
    BP1 = np.zeros((64,))
    pol_gam  = np.ones((30,))
    pol_bet = np.zeros((30,))


    ## VALUE WEIGHTS
    KV0 = np.random.randn(1,256,1,1) * 0.1
    BV0 = np.zeros((1,))
    w0 = np.random.rand(256,64)*0.05
    w1 = np.random.randn(1,256) *0.002
    b0 = np.zeros((256,))
    b1 = np.zeros((1,))

    ### step up convolution

    K_init = np.random.randn(256,20,3,3)
    B_init= np.zeros((256,)) ## steps up so it can eb passed int o rresidual blocks. 

    ### residual block weights## thhese will be stored in one very larg tensor s we can for loop over  it iniices to  acces them, his akes it easie to  store.

    ## each block needs 2x (256,256,3,3) filers and thereofore 2,256  bias evctors, then we need the trianble gamma and beta, we need 2,256 for gamma and 2,256  for beta per block.

    ## so for 12 resiualblocks we will have 

    He = np.sqrt(2/(256*3*3))
    RESID_K = np.random.rand(RESIDUAL_BLOCKS*2,256,256,3,3)* He
    RESID_B = np.zeros((RESIDUAL_BLOCKS*2,256))* small
    RESID_gamma = np.ones((RESIDUAL_BLOCKS*2,256)) * 0.1
    RESID_bet = np.zeros((RESIDUAL_BLOCKS*2,256))





    #for i in range(RESIDUAL_BLOCKS):##intially we see this crazy explosion of the residual blokc avraincce so we start the blck as esteinally an identiy mapping from the strat to the value hea convoution, 
    #    RESID_gamma[2*i + 1, :] = 0.1
else:
    load_full_model("Example_Model.npz")
import numpy as np

def save_full_model(filename="FULL_MODEL.npz"):
    """
    Save policy, value and residual weights/biases/gamma/beta to a compressed .npz file.
    Raises a NameError if any expected variable is not defined in globals().
    """
    keys = [
        "KP0","BP0","KP1","BP1","pol_gam","pol_bet",
        "KV0","BV0","w0","w1","b0","b1",
        "K_init","B_init",
        "RESID_K","RESID_B","RESID_gamma","RESID_bet"
    ]

    # Ensure all variables exist (helpful debug if something is missing)
    missing = [k for k in keys if k not in globals()]
    if missing:
        raise NameError(f"Cannot save model — these variables are not defined: {missing}")

    np.savez_compressed(
        filename,
        KP0=KP0, BP0=BP0,
        KP1=KP1, BP1=BP1,
        pol_gam=pol_gam, pol_bet=pol_bet,
        KV0=KV0, BV0=BV0,
        w0=w0, b0=b0,
        w1=w1, b1=b1,
        K_init=K_init, B_init=B_init,
        RESID_K=RESID_K, RESID_B=RESID_B,
        RESID_gamma=RESID_gamma, RESID_bet=RESID_bet
    )
    print(f"Full model saved to {filename}")

params = [K_init,B_init,KV0,BV0,w0,w1,b0,b1,RESID_gamma,RESID_bet,RESID_B,RESID_K,BP0,BP1,KP0,KP1]

def im2col(input, kernel_size=3, padding=1, stride=1):
    ## this function is simply to vectorise the convolution using matmul as opposed to a simple conv for-loop
    
    C, H, W = input.shape
    pad = padding
    input_padded = np.pad(input, ((0,0), (pad,pad), (pad,pad)), mode='constant') ## make so that 8x8xn shape is upheld
    
    out_h = (H + 2*pad - kernel_size)//stride + 1
    out_w = (W + 2*pad - kernel_size)//stride + 1
    cols = []

    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            patch = input_padded[:, i:i+kernel_size, j:j+kernel_size] ## pulls out patch to be convolved.
            cols.append(patch.reshape(-1))  
    
    cols = np.array(cols).T   # shape (C*k*k, out_h*out_w)
    return cols

def conv(tensor,kernels,bias,k,p):
    cols0 = im2col(tensor, kernel_size=k, padding=p)    # (C*9, 64) 9-> kernel_size**2 64, size_of_input -> number of squartes conv passes over
    K0_mat = kernels.reshape(kernels.shape[0], -1)       # (num_of_input_channels, C*9)
    #
    z0 = K0_mat @ cols0 + bias[:, None]                     
    z0 = z0.reshape(kernels.shape[0], 8, 8)
    return z0 ## raw logits after convolution

def residual_block(batch,filter_1,filter_2,bias_1,bias_2,gam_1,gam_2,bet_1,bet_2):
    n,c,h,w = batch.shape

    ## FIRST PASS ##
    output_1 = np.zeros((batch.shape))## assuming  we have 8x8 --> 8x8, true for this project

    for num in range(n):
        output_1[num] = conv(batch[num],filter_1,bias_1,3,1) ## matching the output of eahc convution to a 4D plane in the output tensor, 
    
    ## now output is like a n list of 3_d output tensors, so shape of... batc size,channels(no of kernels),widht,heiight
    
    ## generate batch statistics, ## this part helped with by chat as batch normalisation was new at the time. 
    mu_1 = np.mean(output_1, axis=(0, 2, 3))  # shape: (C,) ## mean of every channel
    sig2_1 = np.var(output_1, axis=(0, 2, 3)) #shape(C,) ### vraince of each cannnel

    #### BATCH NORMALISATION
    epsilon = 1e-5
    BN_1 = (output_1 - mu_1[None, :, None, None]) / np.sqrt(sig2_1[None, :, None, None] + epsilon)


    ##  appply out = y*z + B

    OUT_1 = gam_1[None, :, None, None] * BN_1 + bet_1[None, :, None, None]


    ## APPLY RELU ## 

    rel_out_1= np.maximum(0,OUT_1)
    
                #### SECOND PASS######

    output_2 = np.zeros((batch.shape))## assuming  we have 8x8 --> 8x8

    for num in range(n):
        output_2[num] = conv(rel_out_1[num],filter_2,bias_2,3,1) ## matching the output of eahc convution to a 4D plane in the output tensor, 
    
    ## now output is like a n list of 3_d output tensors, so shape off... batc size,channels(no of kernels),widht,heiight
    

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

    cache ={
            "x": batch,                
            "s1": output_1, "b1": BN_1, "a1": rel_out_1,
            "s2": output_2, "b2": BN_2,
            "u": SKIP, "y": rel_out_2,
            "var_1": sig2_1, "var_2": sig2_2,
            "mu_1":mu_1,"mu_2":mu_2,
            "OUT_1":OUT_1,"OUT_2":OUT_2,
            "mask_1":(OUT_1>0),"mask_2":(SKIP>0)
            ,"eps":epsilon
                        } 

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




    return rel_out_2,cache

def col2im(cols, input_shape, kernel_size=3, padding=1, stride=1):
    C, H, W = input_shape
    pad = padding
    H_p, W_p = H + 2*pad, W + 2*pad
    input_padded = np.zeros((C, H_p, W_p))
    out_h = (H + 2*pad - kernel_size)//stride + 1
    out_w = (W + 2*pad - kernel_size)//stride + 1

    
    cols = cols.T.reshape(out_h*out_w, C, kernel_size, kernel_size) ## takes the vectorised output of (C*9,64) and reshapes to (C,8,8)
    idx = 0
    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            input_padded[:, i:i+kernel_size, j:j+kernel_size] += cols[idx]
            idx += 1
    return input_padded[:, pad:pad+H, pad:pad+W] ## this is the new tesnor reayd to be passed to next block/stage

def del2del(batch, x, W, kernel_size=3, padding=1): ## this is trakcing gradient though a single convolution step,
    C_out, C_in, k, _ = W.shape
    H, W_in = x.shape[1], x.shape[2]
    nums= batch.shape[0]
    #cols = im2col(x, kernel_size, padding) 
    N = H * W_in


    dw = np.zeros_like(W)
    db = np.zeros((C_out,))
    result = np.zeros((batch.shape[0],C_in,8,8))
   
    for num in range(nums):
        dout = batch[num]

        ## the next 10 - lines were mostly written by LLM 
        cols = im2col(x[num], kernel_size, padding)   
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
  
def step_up_conv(input,K_init,B_init): ## takes input of borad and steps up into 256 channels to be passed into residual trunk
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
    
    return out,cache

def POL_FORWARD(batch,KP0,BP0,KP1,BP1,FENS): ##onlyone layer that is batch normalised. 

    ## FENS are the associated FENS, for eac memebr of thr btch, so that our legal move mask can be applied.  

    ## batch contains the batch outputs after the residual blocks

    ## thhen we convolve to reducte to 30x8x8 each memebr of batch, normalise as norma, then apply, another ocnution to get final 73x8x8 tenosr
    ## then we apply our legal moves mask, then apply our softmax. 
    n,c,h,w = batch.shape
    a,b,d,e = KP0.shape
    output_1 = np.zeros((n,a,h,w))
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
        
        legal_mask = np.where(legal_moves ==1,z,LARG_NEG) ## where tensor =1  replace with conv output if not make it -Larg__neg soo   soft max asigns it small number

        soft = softmax(legal_mask.reshape(-1), axis=0).reshape(64, 8, 8) ## softmax over all legal moves ### could n future add illegal move penalty

        FINAL[num] = soft
        
    cache= {'trunk_out':batch,
            'zconv':output_1,
            'a1':rel_out_1,
            'policy_pred':FINAL

    }




    return FINAL, cache

    ### legal move mask

def VAL_FORWARD_2(batch,KV0,BV0,w0,b0,w1,b1):
    n,c,h,w = batch.shape
    CONV = np.zeros((n,1,h,w)) ## as it is a 1x1 convution we van mapp every trunksample output to a plane of  8x8 boards


    for num in range(n):
        CONV[num]  = conv(batch[num],KV0,BV0,1,0)[0,:,:] ## 1x1 conv soo no paddig neccesary
    
    ## now we have a nx8x8 tensor of results,
    
    #so now we have a tensor of nx8x8 continag all the reuslts of ech sample, we can do all of the  
    
    x_flat = CONV.reshape(n,-1) ## reshapes all boards into n 64 evtcors,this way we can porcess baath in one 

    ## sya w0 is (256,64) and b0 = (256,)
    
    z0 = x_flat@w0.T + b0[None,:]## this process all the smaples thorugh the network in one. 
    

    z1 = z0@w1.T + b1[None,:]
    a1 = np.tanh(z1)

    ## this way we havve strored all smaples simulatenously thorugh and stord all the pre RELU actvation nicely in a matrix,
    ## this kidn of optimatsation is porbably able to be done thoughot the forward prop,however to do this with the forward tensors woud be above  my paygrade
    ## i wa=ould at that point have to devote lareg time to undertnd or just copy down someting I dont undertand.  

    cache = {'trunk_input':batch,
             'zconv':CONV,
             'actConv':CONV,
             'ANN_input':x_flat,
             'z0':z0,'a0':z0,
             'z1':z1,'a1':a1}

    return a1,cache

def del2del_resid(batch, x, W, kernel_size=3, padding=1): # this is specific for the batch back prop through residual blocks as we assume dimenison stays constant
    n,c,h,w = batch.shape
    dw = np.zeros_like(W)
    dB = np.zeros((c,))
    result = np.zeros_like(batch)
    
    C_out, C_in, k, _ = W.shape
    H, W_in = x.shape[1], x.shape[2]

    
    N = H * W_in
    dW = np.zeros_like(W)
    dB = np.zeros((C_out,))
    result = np.zeros((batch.shape[0],C_in,8,8))

    for sample in range(n):
        dout = batch[sample]
        cols = im2col(x[sample], kernel_size, padding)  
        dout_flat = dout.reshape(C_out, -1)   

        dW_ = dout_flat @ cols.T
        dW += dW_.reshape(W.shape)

        dB += np.sum(dout_flat, axis=1)

        W_flat = W.reshape(C_out, -1)
        dcols = W_flat.T @ dout_flat
        result[sample] = col2im(dcols, x[sample].shape, kernel_size, padding)


  
    Nbatch = batch.shape[0]

    dW /= float(Nbatch)
    dB /= float(Nbatch)

    return dB, dW, result

def residual_back(cache,gy,gamma_2,gamma_1,K2,K1):


    mask2 = cache["mask_2"] ## this is RELU mask
    #gy is the upstream graidnt, being passed from previous residual block

    gu = gy*mask2

    g_x_skip = gu.copy()
    g_b2 = gu.copy()

    g_bet_2 = g_b2.sum(axis = (0,2,3)) ## change in beta weights, sum over all of eahc plane

    x_hat_2 = cache["b2"]

    g_gam_2 = (g_b2*x_hat_2).sum(axis = (0,2,3)) ## change in gamma weights

    ### now we look at error in otu from  2nd convolution
    
    d_x_hat_2 = g_b2 * gamma_2[None,:,None,None] # usde in next sum 

    M = g_b2.shape[0]*g_b2.shape[2]*g_b2.shape[3]
    var_2= cache["var_2"]
    eps = cache["eps"]
    pre_factor = 1/(M*np.sqrt(var_2+eps))
    pre_factor = pre_factor[None,:,None,None]
    
    term1= M*d_x_hat_2
    term2= d_x_hat_2.sum(axis= (0,2,3))[None,:,None,None]
    term3 = np.sum(d_x_hat_2*x_hat_2, axis = (0,2,3))

    d_s2 = pre_factor*(term1-term2-x_hat_2*term3[None,:,None,None])  ## this is the 'grad/error' in the convolution 2,


###now we backcnvolve, 
    x = cache["a1"]

    g_bias2,g_K2,g_a1 = del2del_resid(d_s2,x,K2,kernel_size=3,padding=1)


## now we repeat.

    mask1 = cache["mask_1"]
    #gy is th upstream graidnt, being passed form previosu ressudla bock

    g_b1 = g_a1*mask1

    g_bet_1 = g_b1.sum(axis = (0,2,3)) ## change in beta weights

    x_hat_1 = cache["b1"]

    g_gam_1 = (g_b1*x_hat_1).sum(axis = (0,2,3)) ## change in gamma weights

    ### now we look at error in otu from  2nd convolution
    
    d_x_hat_1 = g_b1 * gamma_1[None,:,None,None] # usde in next sum 

    M = g_b1.shape[0]*g_b1.shape[2]*g_b1.shape[3]
    var_1= cache["var_1"]
    eps = cache["eps"]
    pre_factor = 1/(M*np.sqrt(var_1+eps))
    pre_factor = pre_factor[None,:,None,None]

    term1= M*d_x_hat_1
    term2= d_x_hat_1.sum(axis= (0,2,3))[None,:,None,None]
    term3 = np.sum(d_x_hat_1*x_hat_1, axis = (0,2,3))

    d_s1 = pre_factor*(term1-term2-x_hat_1*term3[None,:,None,None])

    ## back convolve thhroug 1 
    x = cache["x"]

    g_bias1,g_K1,g_input = del2del_resid(d_s1,x,K1,kernel_size=3,padding=1)
   
    g_out = g_input + g_x_skip ## total gradient out of this block

    return g_out,g_bias1,g_K1,g_bet_1,g_gam_1,g_bias2,g_K2,g_bet_2,g_gam_2

def value_back(v_true,v_pred,w1,w0,filter,cache):
    #z2 atcivation after hiddenl layer, this is of shape (N,256) # fr batch size
    #z2,hidden_activations,z1,w1,zconv,conv_input,filter,x_flat
    
    z0 = cache['z0']
    a1 = cache['a1']
    zconv = cache['zconv']
    conv_input = cache['trunk_input']
    x_flat = cache['ANN_input']
    
    

    
    n,_ = v_pred.shape

    v_pred = v_pred.reshape(-1, 1)
    v_true = v_true.reshape(-1, 1)
    
    

    del_v = (v_pred-v_true)*(np.ones(v_pred.shape)-v_pred**2)*(1/n) ### this is the del_ vetcor across batch
    

    dw2 = del_v.T@a1
    db2 = del_v.sum(axis=0)

    new_del = del_v @ w1 # shape of (N,256)
    ## now new del is change in  post relu atcivations 

    g_h = new_del * (z0>0) ## grainet of hidden layer pre atcivations  

    dw1 = g_h.T@x_flat ## were z1 is th eflattend output form 1x1 convolution
    db1 = g_h.sum(axis = 0)

    g_conv_out = g_h@w0
    
    g_conv_out = g_conv_out.reshape(n,1,8,8)

    g_conv_out = g_conv_out*(zconv>0)

    ### this is now the errors afer the 1x1 convoltional flattenng, so we will now pass this bak through the convolution as we have done previosul
    # this will give us a twnor of hape n,60,8,8 whihch will then be fed into the trunk

    dbias,dker,out = del2del(g_conv_out,conv_input,filter,kernel_size=1,padding=0)


    return out,dbias,dker,db1/n,dw1/n,db2/n,dw2/n

def value_back_2(v_true,v_pred,w1,w0,filter,cache): ## prupose of this is to experiment wiht the killing of neggative values,so I haev removed RELu activations from conv and NN
    #z2 atcivation after hiddenl layer, this is of shape (N,256) # fr batch size
    #z2,hidden_activations,z1,w1,zconv,conv_input,filter,x_flat
    
    z0 = cache['z0']
    a1 = cache['a1']
    zconv = cache['zconv']
    conv_input = cache['trunk_input']
    x_flat = cache['ANN_input']
    

    
    n,_ = v_pred.shape

    v_pred = v_pred.reshape(-1, 1)
    v_true = v_true.reshape(-1, 1)
    
    
    ## this is sech^2 in disguise
    del_v = (v_pred-v_true)*(np.ones(v_pred.shape)-v_pred**2) ### this is the del_ vetcor, this woudl usually be acalar butas we are batch porcessing 
                                                                            ## then it becomes a vector, each dimension being a smpale
     
    ## I am keeping this, but in retrospect this is a poor design choice 21/09/26
    ## this was an attempt at reducting vanishing gradient at tanh stauration.
    for i in range(n):
        if abs(v_pred[i]) >0.99:
            del_v[i] = (v_pred[i]-v_true[i]) ## just so that predictiosn of 1 don tget suck there


    dw2 = del_v.T@a1
    db2 = del_v.sum(axis=0)

    new_del = del_v @ w1 # shape of (N,256)
    ## now new del is change in  post relu atcivations 

    g_h = new_del 

    dw1 = g_h.T@x_flat ## were z1 is th eflattend output form 1x1 convolution
    db1 = g_h.sum(axis = 0)

    g_conv_out = g_h@w0
    
    g_conv_out = g_conv_out.reshape(n,1,8,8)

    
    ### this is now the errors afer the 1x1 convoltional flattenng, so we will now pass this bak through the convolution as we have done previosul
    # this will give us a twnor of hape n,60,8,8 whihch will then be fed into the trunk

    dbias,dker,out = del2del(g_conv_out,conv_input,filter,kernel_size=1,padding=0) ## weight change to 1x1 conv step 


    return out,dbias,dker,db1/n,dw1/n,db2/n,dw2/n

def pol_back(cache, pol_true,pol_pred,KP1,KP0):

    last_conv_input  = cache['a1']
    zconv = cache['zconv']
    trunk_input= cache['trunk_out']

    del_pol =  pol_pred-pol_true

    dBP1,dKP1,dx= del2del(del_pol,last_conv_input,KP1,kernel_size=3,padding=1)

    new_del = dx*(zconv>0)

    dBP0,dKP0,d_trunk = del2del(new_del,trunk_input,KP0,kernel_size=3,padding=1)

    return d_trunk,dBP0,dKP0,dBP1,dKP1

def grad_report():

    ## Completely written by ChatGPT used to diagnose VG and then subsequent netwrok reconfigurtaion

    print("===== GRAD REPORT =====")
    print("max|del_k_init|", np.max(np.abs(del_k_init)))
    print("mean|del_k_init|", np.mean(np.abs(del_k_init)))
    print("max|del_KERNELS|", np.max(np.abs(del_KERNELS)))
    print("mean|del_KERNELS|", np.mean(np.abs(del_KERNELS)))
    print("max|del_RESID_GAM|", np.max(np.abs(del_RESID_GAM)))
    print("max|dvw0|", np.max(np.abs(dvw0)) if 'dvw0' in locals() else None)
    print("max|dvw1|", np.max(np.abs(dvw1)) if 'dvw1' in locals() else None)
    # Update magnitudes if you were to apply LR:
    print("would-update max (kernels) :", np.max(np.abs(del_KERNELS * lr)))
    print("would-update max (initK)  :", np.max(np.abs(del_k_init * lr)))
    print("would-update max (w1)     :", np.max(np.abs(dvw1 * lr)) if 'dvw1' in locals() else None)
    # signal into conv-backprop: per-element d_s2 statistics (if available in cache, else compute in residual_back)
    last = cache[RESIDUAL_BLOCKS-1]
    
        # print a1 that conv sees:
    print("a1 (last block) max/mean/std:", np.max(np.abs(last['a1'])), np.mean(np.abs(last['a1'])), np.std(last['a1']))
    print("=======================")

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

        ## Note ## Although I understand theory, this function was written by LLM at the time
        
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Initialize first and second moment estimates (same shape as params)
        self.m = [np.zeros_like(p) for p in params]  # momentum (mean grad)
        self.v = [np.zeros_like(p) for p in params]  # variance (mean grad^2)

        self.t = 0  # time step

    def step(self, grads):
        """
        grads: list of numpy arrays (same shape as self.params),
               containing gradients for each parameter
        """
        self.t += 1
        new_params = []

        for i, (p, g) in enumerate(zip(self.params, grads)):
            # Update biased first moment (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Update biased second raw moment (variance)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Parameter update
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p -= update

            new_params.append(p)

        return new_params

# Make Adam optimizer
adam = Adam(params, lr=1e-4)

data = np.load('PROC_FENS_1_SHUFF_LARGE.npy',allow_pickle=True)
data = data.tolist()
len =len(data)

#random.shuffle(data)

batchsize = 50
EPOCHS = 3
lr = 1e-4
Ad = True

batch = np.zeros((batchsize,20,8,8))
for epoch in range(EPOCHS):
    
    for j in range(len//batchsize):
        new_data = data[j*batchsize:(j+1)*batchsize]
        v_true = np.zeros(batchsize,)
        cache = {}
        FENS = []
        pol_target = np.zeros((batchsize,64,8,8))
        #batch preperation
        for i in range(batchsize):
            raw_eval = new_data[i][1]
            turn = new_data[i][0].split(' ')[1]

            if turn == 'b':
                raw_eval = - raw_eval
            
            if abs(raw_eval) == 40: ## this is our internal checkmate value
                v_true[i] = np.sign(raw_eval)
            else:
                v_true[i] = np.tanh(raw_eval/10)

            batch[i] = F.fen_to_tensor(new_data[i][0],turn)

            ## adding ppostions so we can msk in pol forward
            FENS.append(new_data[i][0])
            
            for id,score in new_data[i][2]: ## retirveing the moves and scores
                pol_target[i] += ID_dict[id,turn]*score
        
            
        ##forward pass
        input,init_cache = step_up_conv(batch,K_init,B_init)

        for k in range(RESIDUAL_BLOCKS): ## pass inputs though residual trunk
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

            input, cur_cache = residual_block(input,k_first,k_second,bias_first,bias_second,gam_1,gam_2,bet_1,bet_2) ## add the output to the input of next loop 
            cache[k] = cur_cache ## add thsotred activations to the chache dictionary


        ## both val and pol heds
        v_pred,val_cache = VAL_FORWARD_2(input,KV0,BV0,w0,b0,w1,b1)
        pol_pred, pol_cache =POL_FORWARD(input,KP0,BP0,KP1,BP1,FENS)


        ## POLICY LOSS
        eps = 1e-6 ## to aviod log 0 
        
        safe_pol = np.clip(pol_pred,eps,1)

        neg_log = -np.log(safe_pol)
                ## elemtisw mutiplywith target tensor
        loss_per_example = np.sum(pol_target * neg_log, axis=(1,2,3))  # sum over actions use cross entropy
        L_pol = np.sum(loss_per_example)/batchsize

        
        
        ## back prop
        gy_val,dvbias,dvker,dvb0,dvw0,dvb1,dvw1 = value_back_2(v_true,v_pred,w1,w0,KV0,val_cache)
        gy_pol,dBP0,dKP0,dBP1,dKP1 = pol_back(pol_cache,pol_target,pol_pred,KP1,KP0)

        
        ###combining head graidents
        #lam = np.linalg.norm(gy_val)/np.linalg.norm(gy_pol)
        lam= 0.3 ## shift to weight vlaue head vs pol head graidents. 
        
        gy = (1-lam)*gy_val + lam*gy_pol



        ## gy is thr passing around gradient.        
        del_KERNELS = np.zeros_like(RESID_K)
        del_RESID_BIAS = np.zeros_like(RESID_B)
        del_RESID_BET = np.zeros_like(RESID_bet)
        del_RESID_GAM = np.zeros_like(RESID_gamma)
        for k in range(RESIDUAL_BLOCKS-1,-1,-1): ## worl backwoards for the last block, 

            a = 2*k
            b = 2*k + 1
            cur_cache = cache[k]

            gam_1 = RESID_gamma[a]
            gam_2 = RESID_gamma[b]
            K1 = RESID_K[a]
            K2 = RESID_K[b]

            gy,del_RESID_BIAS[a],del_KERNELS[a],del_RESID_BET[a],del_RESID_GAM[a],del_RESID_BIAS[b],del_KERNELS[b],del_RESID_BET[b],del_RESID_GAM[b] = residual_back(cur_cache,gy,gam_2,gam_1,K2,K1)

        relu_mask = init_cache["relu_mask"]
        input = init_cache["input"]
        gy = gy*relu_mask

        del_bias_init,del_k_init,_ = del2del(gy,input,K_init,kernel_size=3,padding=1)
       

        averager = 1/batchsize



        #grad_report()




#K_init,B_init,KV0,BV0,w0,w1,b0,b1,pol_gam,pol_bet,RESID_gamma,RESID_bet,RESID_B,RESID_K
        if Ad:
            param_grad = [del_k_init,del_bias_init,dvker,dvbias,dvw0,dvw1,dvb0,dvb1,del_RESID_GAM,del_RESID_BET,del_RESID_BIAS,del_KERNELS,dBP0,dBP1,dKP0,dKP1]

            adam.step(param_grad)
        else:
            RESID_K -= del_KERNELS * lr
            RESID_B -= del_RESID_BIAS * lr
            RESID_bet -= del_RESID_BET * lr
            RESID_gamma -= del_RESID_GAM * lr
            K_init -= del_k_init * lr
            B_init -= del_bias_init * lr
            BV0-= dvbias * lr
            KV0 -= dvker * lr
            w0 -= dvw0 * lr
            w1 -= dvw1 * lr
            b0 -= dvb0 * lr
            b1 -= dvb1 * lr
            KP0 -= dKP0 * lr
            KP1 -=dKP1 * lr
            BP0 -=dBP0 * lr
            BP1 -=dBP1 * lr

        v_true = v_true.reshape(-1,1)        
        
        
        for i in range(10):
            print(v_true[i],v_pred[i])

        print("value_loss",((v_pred-v_true)**2).sum(axis = 0)/batchsize)
        print("policy_loss",L_pol)
        print()

        if j%10== 0 and j!=0:
            save_full_model(filename="FULL_MODEL_0")

