import sys

xs = [line.split(',') for line in sys.stdin.read().split('\n')
                      if line.strip() != '']
for attr_id in range(len(xs[0])):
  attr_vals = set(x[attr_id] for x in xs)
  mapping = {}
  for k in attr_vals:
    mapping[k] = str(len(mapping))
  for row in range(len(xs)):
    xs[row][attr_id] = mapping[xs[row][attr_id]]

for x in xs:
  print ','.join(x)
