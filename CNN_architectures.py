import torch

class StreaksCNN(torch.nn.Module):
    
    #Our batch shape for input x is (1, winSz, winSz)
    
    def __init__(self, winSz, filtSz, convPad):
        super(StreaksCNN, self).__init__()
                
        # We use the same pooling for all hidden layers
        self.cnn_part = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=filtSz, stride=1, padding=convPad),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.BatchNorm2d(8),

            torch.nn.Conv2d(8, 16, kernel_size=filtSz, stride=1, padding=convPad),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.BatchNorm2d(16),

            torch.nn.Conv2d(16, 32, kernel_size=filtSz, stride=1, padding=convPad),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, kernel_size=filtSz, stride=1, padding=convPad),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32)
        )

        self.dropout = torch.nn.Dropout2d(0.2) 
                
        #1152 input features, 2 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(6 * 6 * 32, 2)
                
    def forward(self, x):
        x = self.cnn_part(x)
        x = self.dropout(x)
        
        #Size changes from (32, 6, 6) to (1, 1152)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 6 * 6 * 32)
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 1152) to (1, 2)
        x = self.fc1(x)
        
        return(x)
