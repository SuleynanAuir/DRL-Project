import torch.nn as nn

class QModel(nn.Module):
    def __init__(self, in_features=128, num_actions=18):
        '''
        Architecture of Q-function
        :param in_features: dimension of input
        :param num_actions: dimension of output
        '''
        '''
        TODO: Define the architecture of the model here. 
        You may find nn.Sequential helpful.
        '''
        super(QModel, self).__init__()
        self.model = None

        ##############################################################
        ################ YOUR CODE HERE - 5-6 lines ##################
        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        ##############################################################
        ######################## END YOUR CODE #######################

    def forward(self, x):
        return self.model(x)
