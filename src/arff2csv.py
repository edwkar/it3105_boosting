import sys

with open(sys.argv[1], 'r') as arff_file:
  with open(sys.argv[2], 'w') as csv_file:
    lines = arff_file.read().split('\n')

    while lines[0].strip().lower() != '@data':
      lines.pop(0)
    lines.pop(0)

    xs = [line.split(',') for line in lines if line.strip() != '']
    for attr_id in range(len(xs[0])):
      attr_vals = set(x[attr_id] for x in xs)
      mapping = {}
      for k in attr_vals:
        mapping[k] = str(len(mapping))
      for row in range(len(xs)):
        xs[row][attr_id] = mapping[xs[row][attr_id]]

    for x in xs:
      csv_file.write('%s\n' % ','.join(x))
