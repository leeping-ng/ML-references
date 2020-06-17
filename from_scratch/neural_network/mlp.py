import numpy as np

class MLPTwoLayers:

    # DO NOT adjust the constructor params
    def __init__(self, input_size=3072, hidden_size=100, output_size=10, alpha=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha

        # for weights, times 2 and minus 1 so that outputs range from -1 to 1
        # w1 shape: (3072, 100)
        self.w1 = (2*np.random.random((input_size, hidden_size)) - 1)*0.1
        #self.w1 = (np.random.random((input_size, hidden_size)))*np.sqrt(2.0/hidden_size)
        # w2 shape: (100, 10)
        self.w2 = (2*np.random.random((hidden_size, output_size)) - 1)*0.1
        #self.w2 = (np.random.random((hidden_size, output_size)))*np.sqrt(2.0/output_size)
        # b1 shape: (100, 1)
        self.b1 = np.zeros((hidden_size,1))
        # b2 shape: (10, 1)
        self.b2 = np.zeros((output_size,1))


    def forward(self, features):
        """
            Takes in the features
            returns the prediction
        """
        # features shape: (3072,1)
        self.features = features.reshape(self.input_size,1)

        # z_1 shape: (100,1) = (100,3072)*(3072,1)
        self.z_1 = np.matmul(self.w1.T, self.features) + self.b1
        # a_1 shape: (100,1)
        self.a_1 = self.sigmoid(self.z_1)

        # z_2 (logits) shape: (10,1) = (10,100)*(100,1)
        self.z_2 = np.matmul(self.w2.T, self.a_1) + self.b2
        # a_2 (probabilities) shape: (10,1)
        self.a_2 = self.softmax(self.z_2)

        return self.a_2


    def loss(self, preds, label):
        """
            Takes in the predictions and label
            returns the training loss
        """
        # label shape: (10,1)
        self.label = label.reshape(self.output_size,1)
        # ce is a float = np.sum((1,10)*(10,1))
        self.ce = -np.sum(np.matmul(self.label.T,np.log(preds)))

        return self.ce

    def backward(self, train_loss):
        """
            Takes in the loss and adjusts the internal weights/biases
            https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        """
        ##### This section is for layer 2 #####
        # dL_dz2 shape: (10,1)
        # dL_dz2 = dL_da2*da2_dz2, but can be simplified to a_2 - y (see link)
        dL_dz2 = self.a_2 - self.label
        # dz2_dw2 shape: (100,1)
        dz2_dw2 = self.a_1
        # dL_dw2 shape: (100,10) = (100,1)*(1,10)
        dL_dw2 = np.matmul(dz2_dw2, dL_dz2.T)
        # dL_db2 shape: (10,1)
        dL_db2 = dL_dz2
        ##### This section is for layer 2 #####

        ##### This section is for layer 1 #####
        # dz2_da1 shape: (100,10)
        dz2_da1 = self.w2
        # da1_dz1 shape: (100,1)
        da1_dz1 = self.d_sigmoid(self.z_1)
        # dz1_dw1 shape: (3072,1)
        dz1_dw1 = self.features
        ##### This section is for layer 1 #####

        ##### This section to get dL_dw1 #####
        # dL_da1 shape: (100,1) = (100,10)*(10,1)
        dL_da1 = np.matmul(dz2_da1, dL_dz2) #switched order, doesn't matter
        # dL_dz1 shape: (100,1)
        dL_dz1 = np.multiply(dL_da1, da1_dz1)
        # dL_dw1 shape: (3072,100)
        dL_dw1 = np.matmul(dz1_dw1, dL_dz1.T)
        # dL_db1 shape: (100,1)
        dL_db1 = dL_dz1
        ##### This section to get dL_dw1 #####

        self.w1 -= self.alpha * dL_dw1
        self.w2 -= self.alpha * dL_dw2
        self.b1 -= self.alpha * dL_db1
        self.b2 -= self.alpha * dL_db2
       

    def softmax(self, x):
        """
            Takes in array of logits
            Returns array of probabilities by applying softmax
        """
        # try numerically stable softmax
        return np.exp(x) / np.sum(np.exp(x))

    def sigmoid(self, x):
        """
            Takes in array and applies sigmoid function
        """
        return 1.0/(1 + np.exp(-x))
    
    def d_sigmoid(self, x):
        """
            Takes in array and applies derivative of sigmoid function
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))






