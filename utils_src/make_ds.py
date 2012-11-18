from random import randint
from itertools import product

def make_ds():
  for t in product(xrange(4), xrange(3), xrange(2), xrange(2), xrange(2)):
    print ','.join(map(str, list(t)) + [str(randint(0, 2))])

make_ds()

