# CsiNet-Pytorch  
Pytorch implementation of CsiNet (C. Wen, 2018)  
and online learning (Y. Cui, 2022)

Chao-Kai Wen, Wan-Ting Shih, and Shi Jin, “Deep learning for massive MIMO CSI feedback,” IEEE Wireless Communications Letters, 2018.   
Yiming Cui, Jiajia Guo, Chao-Kai Wen, Shi Jin, and Shuangfeng Han, "Unsupervised Online Learning in Deep Learning-Based Massive MIMO CSI Feedback", Unsupervised Online Learning in Deep Learning-Based Massive MIMO CSI Feedback, 2022.

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


**update 20230116**  
Here is online learning with different scenario added.  
**What is online learning?**  
In the real time communication scenario, each BS experience different environment (i.e., different probability density of channel).    
Therefore, it is helpful to do fine tuning with each BS's data.  
Since UE cannot transmit the whole channel matrix to BS,   
BS doesn't have knowledge about data label.  
Moreover, BS can't deliver the gradient flow to UE, so that the encoder in UE cannot be updated.  
Therefore, the online learning only learns decoder part.  


