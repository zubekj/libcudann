import pycudann

train_set = pycudann.load_learning_set('../datasets/mushroom.train')
test_set = pycudann.load_learning_set('../datasets/mushroom.test')

# Activation function: SIGM or TANH
net = pycudann.FeedForwardNN([train_set.getNumOfInputsPerInstance(), 30,
                              train_set.getNumOfOutputsPerInstance()],
                             [pycudann.SIGM, pycudann.SIGM, pycudann.SIGM])

trainer = pycudann.FeedForwardNNTrainer()
trainer.selectNet(net)
trainer.selectTrainingSet(train_set)
trainer.selectTestSet(test_set)

# Default parameters:
# trainer.train(device=pycudann.TRAIN_GPU, algorithm=pycudann.ALG_BATCH, learning_rate=0.7,
#               momentum=0.0, max_epochs=1000, desired_error=0.001,
#               epochs_between_reports=100, shuffle=True,
#               error_function=pycudann.ERROR_TANH, print_type=pycudann.PRINT_MIN)
# device: TRAIN_GPU or TRAIN_CPU
# algorithm: ALG_BATCH or ALG_BP
# error_function: ERROR_TANH or ERROR_LINEAR
# print_type: PRINT_OFF, PRINT_MIN or PRINT_ALL

print(trainer.train(device=pycudann.TRAIN_GPU, print_type=pycudann.PRINT_ALL, max_epochs=1500,
                epochs_between_reports=500, desired_error=0.002))


# Alternative interface (in scikit-learn form):
X = [[1,1],[1,0],[0,1],[0,0]]
Y = [2,2,2,0]
clf = pycudann.NeuralNetClassifier()
clf.fit(X,Y)
print clf.predict([1,1])
print clf.predict_proba([1,1])
print clf.predict_proba([1,0])
print clf.predict_proba([0.5,0])
print clf.predict_proba([0,0])
