import numpy as np
from mlp import MLP
import urllib.request

# Download from given online dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
filename = "letter-recognition.data"
urllib.request.urlretrieve(url, filename)

# load the data
data = []
with open(filename, 'r') as file:
    for line in file:
        values = line.strip().split(',')
        # change letters to numerical values 0-25
        label = ord(values[0]) - ord('A')
        # change string to integers
        values = np.array(list(map(int, values[1:])))
        data.append((values, label))

# setinputs and targets
inputs = np.array([entry[0] for entry in data])
targets = np.array([entry[1] for entry in data])

num_samples = len(inputs)
# 4/5 for training
train_samples = int(num_samples * 0.8)

#split into train and test set
train_inputs, test_inputs = inputs[:train_samples], inputs[train_samples:]
train_targets, test_targets = targets[:train_samples], targets[train_samples:]

# parameters
NI = 16
NO = 26
NH = 20
learning_rate = 0.01
epochs = 1000


mlp = MLP(NI, NH, NO)
#train
for epoch in range(epochs):
    total_error = 0
    for i in range(len(train_inputs)):
        output = mlp.forward(train_inputs[i])
        target = np.zeros(NO)
        target[train_targets[i]] = 1
        total_error += mlp.backward(train_inputs[i], target)
        mlp.update_weights(learning_rate)
    
    # Print loss at intervals
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss - {total_error:.4f}')

# test and write predictions to a file
with open('special_test_outputs.txt', 'w') as file:
    correct = 0
    for i in range(len(test_inputs)):
        output = mlp.forward(test_inputs[i])
        predicted = np.argmax(output)
        file.write(f"Input: {test_inputs[i]} -> Predicted Output: {predicted}, Actual Target: {test_targets[i]}\n")
        
        if predicted == test_targets[i]:
            correct += 1

    accuracy = correct / len(test_inputs)
    file.write(f'Test Accuracy: {accuracy * 100:.2f}%')
