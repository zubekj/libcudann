import pycudann

train_set = pycudann.load_learning_set('../datasets/mushroom.train')
test_set = pycudann.load_learning_set('../datasets/mushroom.test')

net = pycudann.FeedForwardNN([train_set.getNumOfInputsPerInstance(), 30,
                              train_set.getNumOfOutputsPerInstance()],
                             [pycudann.SIGM, pycudann.SIGM, pycudann.SIGM])

trainer = pycudann.FeedForwardNNTrainer()
trainer.selectNet(net)
trainer.selectTrainingSet(train_set)
trainer.selectTestSet(test_set)
print trainer.train(print_type=pycudann.PRINT_ALL, max_epochs=1500,
                epochs_between_reports=500, desired_error=0.002)
