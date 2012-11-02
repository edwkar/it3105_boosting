from random import *

for r in range(100):
  print ','.join(str(randint(1, k)) for k in [2, 2, 2, 2, 2, 3, 2, 2, 4, 4, 2])
