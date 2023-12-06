import numpy as np
from mlp import MLP

# Generate 500 vectors with random components between -1 and 1
input_vectors = np.random.uniform(-1, 1, size=(500, 4))

# calculate sin(x1-x2+x3-x4)
output_values = np.sin(input_vectors[:, 0] - input_vectors[:, 1] + input_vectors[:, 2] - input_vectors[:, 3])

#split dataset into 400 trains and 100 tests
train_input = input_vectors[:400]
train_output = output_values[:400]

test_input = input_vectors[400:]
test_output = output_values[400:]

#parameters
NI = 4
# try different hidden size from 5,6,7,8,9
NH = 5
NO = 1
learning_rate = 0.1
epochs = 1000

mlp = MLP(NI, NH, NO)
with open('sin_errors.txt', 'w') as file:
    # train
    for epoch in range(epochs):
        train_error = 0
        for i in range(len(train_input)):
            output = mlp.forward(train_input[i])
            train_error += mlp.backward(train_input[i], train_output[i])
            mlp.update_weights(learning_rate)
        file.write(f'Epoch {epoch}: Training Loss - {train_error:.4f}\n')
        #  print training loss of every 100 epoch
        if epoch % 100 == 0:
           print(f'Epoch {epoch}: Training Loss - {train_error:.4f}') 

    # Test
    test_loss = 0
    for i in range(len(test_input)):
        prediction = mlp.forward(test_input[i])
        test_loss += np.square(test_output[i] - prediction)
    test_loss /= len(test_input)
    test_loss_scalar = test_loss.item()
    file.write(f'Test Loss: {test_loss_scalar:.4f}\n')
    print(f'Test Loss: {test_loss_scalar:.4f}')
