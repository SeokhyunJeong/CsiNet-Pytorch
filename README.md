# CsiNet-Pytorch  
Pytorch implementation of CsiNet (C. Wen, 2018)  

Chao-Kai Wen, Wan-Ting Shih, and Shi Jin, “Deep learning for massive MIMO CSI feedback,” IEEE Wireless Communications Letters, 2018.   

some of functions are from https://github.com/WilliamYangXu/CSITransformer.
  
**data**:  
You can download QuaDRiGa in https://quadriga-channel-model.de/ and generate channel.  
      Please put the channel data csv file in ./filepath folder.  
      2000 sample channel data (5.3GHz) are given in ./filepath  
      
**run**:   
open 'main.py'  
adjust parameters (dimension of codeword, the number of epochs, ...)  
     and run 'main.py'.  
     
  
**explanation of each python file**  
- data.py: generates data, returns train dataloader and test dataloader  
- train.py: run train epochs and evaluations.  
- model.py: neural network models of encoer and decoder  
- main.py: run file  
