from neuron import Layer
if __name__ == "__main__":
    x = [2.0, 3.0]
    neuron_layer = Layer(2, 3)
    print(neuron_layer(x))

