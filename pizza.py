import numpy as np

t = 'T'
m = 'M'

class Pizza(object):

  def __init__(self, pizza):
    self.pslices = []
    self.pizza = np.array(pizza)
    self.pslicepos = np.zeros(self.pizza.shape)
    self.score = 0

  def populatePizza(self):
    cid = 1
    for (x,y), cell in np.ndenumerate(self.pslicepos):
        cell = PSlice(x,y,1,1, self)

  def findNeighbours(self, pslice):
    ngbs = []
    print pslice
    if pslice.y != 0:
      print 1
      print pslice.y-1, pslice.y, pslice.x, pslice.xe
      print self.pslicepos[pslice.y-1:pslice.y, pslice.x:pslice.xe]
      ngbs.append([i for i in np.nditer(self.pslicepos[pslice.x:pslice.xe, pslice.y-1:pslice.y])])
    if pslice.x != 0:
      print 2
      print pslice.y, pslice.ye, pslice.x-1, pslice.x
      print self.pslicepos[pslice.y:pslice.ye, pslice.x-1:pslice.x]
      ngbs.append([i for i in np.nditer(self.pslicepos[pslice.x-1:pslice.x, pslice.y:pslice.ye])])
    if pslice.ye+1 < self.pslicepos.shape[1]:
      print 3
      print pslice.ye+1, pslice.ye+2, pslice.x, pslice.xe
      print self.pslicepos[pslice.ye+1:pslice.ye+2, pslice.x:pslice.xe]
      ngbs.append([i for i in np.nditer(self.pslicepos[pslice.x:pslice.xe, pslice.ye+1:pslice.ye+2])])
    if pslice.xe+1 < self.pslicepos.shape[0]:
      print 4
      print pslice.y, pslice.ye, pslice.xe+1, pslice.xe+2
      print self.pslicepos[pslice.xe+1:pslice.xe+2, pslice.y:pslice.ye]
      ngbs.append([i for i in np.nditer(self.pslicepos[pslice.y:pslice.ye, pslice.xe+1:pslice.xe+2])])
    return ngbs

class PSlice(object):

  nextid = 1

  def __init__(self, x, y, xlen, ylen, pizza):
    self.id = self.nextID()
    self.x = x
    self.y = y
    self.xe = x + xlen
    self.ye = y + ylen
    self.pizza = pizza
    self.neighbours = pizza.findNeighbours(self)

  def __str__(self):
    return "<PSlice id={} x={} y={} xe={} ye={}>".format(self.id, self.x, self.y, self.xe, self.ye)

  def score(self):
    pass

  def nextID(self):
    tmpid = PSlice.nextid
    PSlice.nextid += 1
    return tmpid
