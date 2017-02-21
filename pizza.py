import numpy as np

class Pizza(object):

  def __init__(self, pizza):
    self.pslices = []
    self.pizza = np.array(pizza)
    self.pslicepos = np.zeros(self.pizza.shape)
    self.score = 0

  def populatePizza(self):
    cid = 1
    for (x,y), cell in np.ndenumerate(self.pslicepos):
        cell = PSlice(x,y,1,1)

  def findNeighbours(self, pslice):
    ngbs = []
    if pslice.y != 0:
      ngbs.append([i for i in np.nditer(self.pslicepos[pslice.y-1:pslice.y, pslice.x:pslice.xe])])
    if pslice.x != 0:
      ngbs.append([i for i in np.nditer(self.pslicepos[pslice.y:pslice.ye, pslice.x-1:pslice.x])])
    if pslice.ye < self.pslicepos.shape[1]:
      ngbs.append([i for i in np.nditer(self.pslicepos[pslice.ye+1:pslice.ye+2, pslice.x:pslice.xe])])
    if pslice.xe < self.pslicepos.shape[0]:
      ngbs.append([i for i in np.nditer(self.pslicepos[pslice.y:pslice.ye, pslice.xe+1:pslice.xe+2])])
    return ngbs

class PSlice(object):

  nextid = 1

  def __init__(self, x, y, xlen, ylen):
    self.id = self.nextID()
    self.x = x
    self.y = y
    self.xe = x + xlen
    self.ye = y + ylen
    self.neighbours = Pizza.findNeighbours(self)

  def score(self):
    pass

  def nextID(self):
    tmpid = nextid
    self.nextid += 1
    return tmpid
