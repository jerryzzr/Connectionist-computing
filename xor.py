from mlp import MLP
import numpy as np
# XOR dataset
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# parameters
NI = 2
NH = 3
NO = 1
learning_rate = 0.4 # Try different rate the best is 0.4
epochs = 10000
# create and train the MLP
mlp = MLP(NI, NH, NO)
with open('xor_errors.txt', 'w') as file:
    for epoch in range(epochs):
        output = mlp.forward(x)
        error = mlp.backward(x, y)
        mlp.update_weights(learning_rate)
        file.write(f'Epoch {epoch}: Training Loss - {error:.4f}\n')
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}: Loss - {error:.4f}')

    print("Final predictions:")
    # Print final predictions and save to file
    for i in range(len(x)):
        prediction = mlp.forward(x[i])
        file.write(f"Input: {x[i]} -> Predicted Output: {prediction}\n")
        print(f"Input: {x[i]} -> Predicted Output: {prediction}")