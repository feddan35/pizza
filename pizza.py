import numpy as np
from Queue import Queue

class Pizza(object):

  interact = True

  def __init__(self, pizza, constraints):
    self.pizza = np.array(pizza)
    self.pslices = np.empty(self.pizza.shape, object)
    self.pslices_i = []
    self.mincomp = constraints.mincomp
    self.maxsize = constraints.maxsize

  def populatePizza(self):
    for (x, y), cell in np.ndenumerate(self.pslices):
        pslice = PSlice(x, y, 1, 1, self)
        self.pslices[x, y] = pslice
        self.pslices_i.append(pslice)

  def score(self):
    return sum(map(lambda x: x.score(), self.pslices_i))

  def __str__(self):
    tos = np.chararray(self.pizza.shape, itemsize=10)
    for x in range(self.pizza.shape[0]):
      for y in range(self.pizza.shape[1]):
        tos[x,y] = str(self.pizza[x,y] + str(self.pslices[x,y].id))
    return tos.__str__()

class PSlice(object):

  nextid = 1

  def __init__(self, x, y, xlen, ylen, pizza):
    self.id = self.nextID()
    self.pizza = pizza
    self.x = x
    self.y = y
    self.xe = x + xlen
    self.ye = y + ylen

  def __str__(self):
    return "<PSlice id={} x={} y={} xe={} ye={}>".format(self.id, self.x, self.y, self.xe, self.ye)

  def __rep__(self):
    return "<PSlice id={} x={} y={} xe={} ye={}>".format(self.id, self.x, self.y, self.xe, self.ye)

  def score(self):
    if self.satisfiesConstraints():
      return self.size()
    else:
      return 0
     
  def vals(self):
    return [i for i in self.pizza.pizza[self.x:self.xe, self.y:self.ye].flatten()]
 
  def satisfiesConstraints(self):
    return min(self.nmashrooms(), self.ntomatoes()) >= self.pizza.mincomp and self.size() <= self.pizza.maxsize

  def ntomatoes(self):
    return len([i for i in self.vals() if i == t])

  def nmashrooms(self):
    return len([i for i in self.vals() if i == t])

  def size(self):
    return (self.xe - self.x) * (self.ye - self.y)

  def nextID(self):
    tmpid = PSlice.nextid
    PSlice.nextid += 1
    return tmpid

  def findNeighbours(self):
    return self.ngbs_left() + self.ngbs_right() + self.ngbs_down() + self.ngbs_up()

  def ngbs_right(self):
    if self.ye < self.pizza.pslices.shape[1]:
      return [i for i in self.pizza.pslices[self.x:self.xe, self.ye:self.ye+1].flatten() if not i is None]
    else:
      return []

  def ngbs_down(self):
    if self.xe < self.pizza.pslices.shape[0]:
      return [i for i in self.pizza.pslices[self.xe:self.xe+1, self.y:self.ye].flatten() if not i is None]
    else:
      return []

  def ngbs_up(self):
    if self.x != 0: 
      return [i for i in self.pizza.pslices[self.x-1:self.x, self.y:self.ye].flatten() if not i is None]
    else:
      return []

  def ngbs_left(self):
    if self.y != 0:
      return [i for i in self.pizza.pslices[self.x:self.xe, self.y-1:self.y].flatten() if not i is None]
    else:
      return []


class Constraints(object):

  def __init__(self, mincomp, maxsize):
    self.mincomp = mincomp
    self.maxsize = maxsize

class ExpansiveCutter(object):

  def __init__(self, pizza):
    self.pizza = pizza
    self.q = Queue()

  def cut(self):
    for s in self.pizza.pslices.flatten():
      self.q.put(s)
    while not self.q.empty():
      cs = self.q.get()
      print self.pizza
      if cs.x != 0:
        res = self.expandUp(cs)
        if res:
          continue
      if cs.y != 0:
        res = self.expandLeft(cs)
        if res:
          continue
      if cs.ye < self.pizza.pizza.shape[1]:
        res = self.expandRight(cs)
        if res:
          continue
      if cs.xe < self.pizza.pizza.shape[0]:
        res = self.expandDown(cs)
        if res:
          continue

  def expandUp(self, pslice):
    ngbs = pslice.ngbs_up()
    ngbsscores = sum(map(lambda x: x.score(), ngbs))
    pslicescore = pslice.score()
    fngbsscores = sum(map(lambda x: self.futureScore(x.x, x.y, x.xe-1, x.ye), ngbs))
    fpslicescore = self.futureScore(pslice.x-1, pslice.y, pslice.xe, pslice.ye)
    if fpslicescore - pslicescore > fngbsscores - ngbsscores:
      pslice.x = pslice.x - 1
      self.pizza.pslices[pslice.x:pslice.x+1,pslice.y:pslice.ye].fill(pslice)
      for n in ngbs:
        n.xe = n.xe - 1
        if n.xe == n.x:
          self.pizza.pslices_i.remove(n)
        else:
          self.q.put(n)
      return True
    else:
      return False

  def expandDown(self, pslice):
    ngbs = pslice.ngbs_up()
    ngbsscores = sum(map(lambda x: x.score(), ngbs))
    pslicescore = pslice.score()
    fngbsscores = sum(map(lambda x: self.futureScore(x.x + 1, x.y, x.xe, x.ye), ngbs))
    fpslicescore = self.futureScore(pslice.x, pslice.y, pslice.xe + 1, pslice.ye)
    if fpslicescore - pslicescore > fngbsscores - ngbsscores:
      pslice.xe = pslice.xe + 1
      self.pizza.pslices[pslice.xe:pslice.xe+1,pslice.y:pslice.ye].fill(pslice)
      for n in ngbs:
        n.x = n.x + 1
        if n.xe == n.x:
          self.pizza.pslices_i.remove(n)
        else:
          self.q.put(n)
      return True
    else:
      return False


  def expandLeft(self, pslice):
    ngbs = pslice.ngbs_up()
    ngbsscores = sum(map(lambda x: x.score(), ngbs))
    pslicescore = pslice.score()
    fngbsscores = sum(map(lambda x: self.futureScore(x.x, x.y, x.xe, x.ye-1), ngbs))
    fpslicescore = self.futureScore(pslice.x, pslice.y-1, pslice.xe, pslice.ye)
    if fpslicescore - pslicescore > fngbsscores - ngbsscores:
      pslice.y = pslice.y - 1
      self.pizza.pslices[pslice.x:pslice.xe,pslice.y:pslice.y+1].fill(pslice)
      for n in ngbs:
        n.ye = n.ye - 1
        if n.ye == n.y:
          self.pizza.pslices_i.remove(n)
        else:
          self.q.put(n)
      return True
    else:
      return False


  def expandRight(self, pslice):
    print "expanding {} right".format(pslice.id)
    ngbs = pslice.ngbs_up()
    ngbsscores = sum(map(lambda x: x.score(), ngbs))
    pslicescore = pslice.score()
    fngbsscores = sum(map(lambda x: self.futureScore(x.x, x.y+1, x.xe, x.ye), ngbs))
    fpslicescore = self.futureScore(pslice.x, pslice.y, pslice.xe, pslice.ye+1)
    print "{} > {} ? {}".format(fpslicescore - pslicescore, fngbsscores - ngbsscores, fpslicescore - pslicescore > fngbsscores - ngbsscores)
    if fpslicescore - pslicescore > fngbsscores - ngbsscores:
      pslice.ye = pslice.ye + 1
      self.pizza.pslices[pslice.x:pslice.xe,pslice.ye:pslice.ye+1].fill(pslice)
      for n in ngbs:
        n.y = n.y + 1
        if n.ye == n.y:
          self.pizza.pslices_i.remove(n)
        else:
          self.q.put(n)
      return True
    else:
      return False

  def futureScore(self, x, y, xe, ye):
    p = PSlice(x, y, xe, ye, self.pizza)
    print p
    return p.score()
    

if __name__ == "__main__":
  t = 'T'
  m = 'M'
  cs = Constraints(2, 4)
  p = Pizza([[t,t,t,t],[m,m,m,m],[t,m,t,m]], cs)
  cutter = ExpansiveCutter(p)
  p.populatePizza()
  import code; code.interact(local=locals())
