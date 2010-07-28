
from barycenter import Barycenter

def builder(kind, embedding, n_neighbors, neigh, neigh_alternate_arguments):
    """
    Function that will create a builder depending on the arguments it is passed
    """
    if kind == "Barycenter":
        mapping = Barycenter(n_neighbors = n_neighbors, neigh = neigh,
        neigh_alternate_arguments = neigh_alternate_arguments)
    elif issubclass(kind, object):
        mapping = kind(n_neighbors = n_neighbors, neigh = neigh,
        neigh_alternate_arguments = neigh_alternate_arguments)
    else:
        mapping = kind
    mapping.fit(embedding)
    return mapping
