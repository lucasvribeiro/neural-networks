import torch
import sys
from torch import nn

DATASETS_PATHS = [
  "./datasets/uci/balance-scale.csv", 
  "./datasets/uci/hayes-roth.csv", 
  "./datasets/uci/mammographic-masses.csv",
  "./datasets/kaggle/zoo.csv",
  "./datasets/kaggle/glass.csv",
  "./datasets/kaggle/mobile-price.csv",
  ]

NUMBER_OF_CLASSES = [3, 3, 2, 7, 7, 4]

class NeuralNetwork:
  def __init__(self):
    self.dataset = None
    self.labels = None

  def run(self):
    print("Starting running...")
    for i in range(len(DATASETS_PATHS)):
      print("Loading " + DATASETS_PATHS[i] + "...")
      self.loadDataset(DATASETS_PATHS[i])

      print("Dataset loaded...")
      print("Starting training network...")
      self.trainModel(NUMBER_OF_CLASSES[i])

      print("Finished training!\n")

  def loadDataset(self, path):
    dataset = []
    labels = []

    file = open(path, "r")
    rows = file.read().split("\n")
    file.close()

    for i in range(len(rows)):
      if i == 0 or rows[i] == "":
        continue

      attributes = rows[i].split(",")
      for j in range(len(attributes)):
        if(attributes[j] == "?"):
          attributes[j] = float(attributes[j-1])
        else:
          attributes[j] = float(attributes[j])

      dataset.append(attributes[0:-1])
      labels.append(attributes[-1])
    
    self.dataset = torch.tensor(dataset).float()
    self.labels = torch.tensor(labels).long()

  def trainModel(self, classesNumber):
    entryNumber = len(self.dataset[0])

    networkOne = nn.Sequential(
      nn.Linear(entryNumber, 20),
      nn.Sigmoid(),
      nn.Linear(20, 40),
      nn.Sigmoid(),
      nn.Linear(40, classesNumber),

      nn.Softmax(dim=1)
    )

    networkTwo = nn.Sequential(
      nn.Linear(entryNumber, 100),
      nn.Sigmoid(),
      nn.Linear(100, 200),
      nn.Sigmoid(),
      nn.Linear(200, classesNumber),

      nn.Softmax(dim=1)
    )

    networkThree = nn.Sequential(
      nn.Linear(entryNumber, 20),
      nn.Sigmoid(),
      nn.Linear(20, 40),
      nn.Sigmoid(),
      nn.Linear(40, 80),
      nn.Sigmoid(),
      nn.Linear(80, classesNumber),

      nn.Softmax(dim=1)
    )

    networkFour = nn.Sequential(
      nn.Linear(entryNumber, 20),
      nn.Sigmoid(),
      nn.Linear(20, 40),
      nn.Sigmoid(),
      nn.Linear(40, 80),
      nn.Sigmoid(),
      nn.Linear(80, 40),
      nn.Sigmoid(),
      nn.Linear(40, classesNumber),

      nn.Softmax(dim=1)
    )

    networkFive = nn.Sequential(
      nn.Linear(entryNumber, 100),
      nn.Sigmoid(),
      nn.Linear(100, 200),
      nn.Sigmoid(),
      nn.Linear(200, 250),
      nn.Sigmoid(),
      nn.Linear(250, 150),
      nn.Sigmoid(),
      nn.Linear(150, classesNumber),    

      nn.Softmax(dim=1)
    )

    networks = [networkOne, networkTwo, networkThree, networkFour, networkFive]
    loss = nn.CrossEntropyLoss()
    eta = 0.03

    for network in networks:
      optimizer = torch.optim.SGD(network.parameters(), lr=eta)

      for epoch in range(100):
        cost = 0
        acc = 0

        for i in range(len(self.dataset)):
          optimizer.zero_grad()

          x = self.dataset[i:i+1]
          d = self.labels[i:i+1]
          y = network(x)
          l = loss(y, d)
          l.backward()

          cost += float(l)
          optimizer.step()

          if y[0].argmax() == d[0]:
            acc += 1

        print(f"Epoch = {epoch} | Loss = {l/len(self.dataset)} | Accuracy = {acc/len(self.dataset)}")
      print("------ Finished Topology! ------")

NeuralNetwork().run()



