
from ....neighbors import Neighbors

class NewNeighbors(Neighbors):
  __call__ = Neighbors.kneighbors

  def __init__(self, k, **kwargs):
    Neighbors.__init__(self, k)
  