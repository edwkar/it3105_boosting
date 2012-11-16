import sys

comma_splitted = lambda xs: [x.split(',') for x in xs if x.strip() != '']

def advance(arff_lines):
  while arff_lines[0].strip().lower() != '@data':
    arff_lines.pop(0)
  arff_lines.pop(0)
  return arff_lines

def create_mapping(lines):
  xs = comma_splitted(lines)
  mappings = {}
  for attr_id in range(len(xs[0])):
    attr_vals = set(x[attr_id] for x in xs)
    mapping = {}
    for k in attr_vals:
      mapping[k] = str(len(mapping))
    for row in range(len(xs)):
      xs[row][attr_id] = mapping[xs[row][attr_id]]
    mappings[attr_id] = mapping
  return mappings

def output_mapped(orig_lines, output_file, mappings):
  xs = comma_splitted(orig_lines)
  for row in range(len(xs)):
    for attr_id in range(len(xs[0])):
      xs[row][attr_id] = mappings[attr_id][ xs[row][attr_id] ]
  for x in xs:
    output_file.write('%s\n' % ','.join(x))

arff_train_file = open(sys.argv[1], 'r')
arff_test_file = open(sys.argv[2], 'r')
csv_train_file = open(sys.argv[3], 'w')
csv_test_file = open(sys.argv[4], 'w')

arff_train_lines = advance(arff_train_file.read().split('\n'))
arff_test_lines = advance(arff_test_file.read().split('\n'))

mapping = create_mapping(arff_train_lines + arff_test_lines)
output_mapped(arff_train_lines, csv_train_file, mapping)
output_mapped(arff_test_lines, csv_test_file, mapping)
