from neuron import Layer, MLP
if __name__ == "__main__":
    #we want 3 inputs -> 2 4-hidden layer -> 1 output
    x = [2.0, 3.0, 1.0]
    ml_perceptron = MLP(3,[4,4,1])
    print(ml_perceptron(x))
   
    

