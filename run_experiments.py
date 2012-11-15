from subprocess import check_output
import random
import re

random.seed(0x42)

SIMPLE_SETS = ['glass.arff', 'page-blocks.arff', 'yeast.arff',]
RAW_DATASETS_PATH = './datasets/raw'
WEKA_PATH = './weka.jar'
TRAIN_SPLIT_PERCENT = 80
DANCE_HALL = '/tmp'
BOOSTER_CLASS_FILES = './target/scala-2.9.1/classes'

def run(cmd_line, **extra_kw_args):
  return check_output(cmd_line, shell=True, **extra_kw_args)

def weka(cmd):
  return run('java -Xmx2000M %s' % cmd, env={ 'CLASSPATH': WEKA_PATH })

def report_output(o):
  for line in o.split('\n'):
    print '  >', line

#print '(Re-)Compiling project...'
#report_output(run('sbt compile'))

ds_rand_path = DANCE_HALL + '/.ds.randomized.arff'
train_path = DANCE_HALL + '/.ds.train.arff'
test_path = DANCE_HALL + '/.ds.test.arff'
train_disc_path = DANCE_HALL + '/.ds.test.train.arff'
test_disc_path = DANCE_HALL + '/.ds.test.disc.arff'
train_disc_csv_path = DANCE_HALL + '/.ds.test.train.csv'
test_disc_csv_path = DANCE_HALL + '/.ds.test.disc.csv'

def parse_result(weka_output):
  m = re.match(
    ('^.*?Error on training data.*?Correctly Classified Instances\s+(\d+)\s' +
      '.*?Error on test data.*?Correctly Classified Instances\s+(\d+).*$'),
    weka_output, re.DOTALL | re.MULTILINE)
  #print weka_output
  return int(m.group(1)), int(m.group(2))

all_scores = []

CLASSIFIER_CONFIGURATIONS = (
  ('40dt[maxDepth=200]', 'weka.classifiers.trees.J48 -I 40',),
)

for ds_name in ('glass', 'page-blocks', 'yeast',):
#for ds_name in ('yeast',):
  print 'Processing ds "%s"...' % ds_name

  for edw_classifier_conf, weka_classifier_conf in CLASSIFIER_CONFIGURATIONS:
    scores = { 'edw_train':  [], 'edw_test':  [],
               'weka_train': [], 'weka_test': [] }
    score_key = (ds_name, edw_classifier_conf, weka_classifier_conf,)
    all_scores.append((score_key, scores))

    for i in range(20):
      print 'Starting iteration %d for dataset %s...' % (1+i, ds_name,)
      ds_path = '%s/%s.arff' % (RAW_DATASETS_PATH, ds_name,)

      print '- Randomizing...'
      weka('weka.core.Instances randomize %d %s > %s/.ds.randomized.arff' % (
           random.randint(0, 10000), ds_path, DANCE_HALL,))

      print ('- Splitting into training and test sets (at %d percent)...' %
              TRAIN_SPLIT_PERCENT)
      weka(('weka.filters.unsupervised.instance.RemovePercentage -P %d ' +
            '-i %s -o %s -V') % (TRAIN_SPLIT_PERCENT,
                                 ds_rand_path, train_path,))
      weka(('weka.filters.unsupervised.instance.RemovePercentage -P %d ' +
            '-i %s -o %s') % (TRAIN_SPLIT_PERCENT, ds_rand_path,test_path,))

      print '- Discretizing...'
      weka(('weka.filters.supervised.attribute.Discretize ' +
            '-b -i %s -o %s -r %s -s %s -Y -c last') % (
            train_path, train_disc_path,
            test_path, test_disc_path,))

      print '- Massaging sets to restricted .csv form...'
      for i, o in ((train_disc_path, train_disc_csv_path,),
                   (test_disc_path, test_disc_csv_path,)):
        run('python src/arff2csv.py %s %s' % (i, o,))


      print '- Running AdaBoost classifier with configuration "%s"...' % (
             edw_classifier_conf,)
      edw_output = run('scala -cp %s Main %s %s %s' % (
                       BOOSTER_CLASS_FILES,
                       train_disc_csv_path, test_disc_csv_path,
                       edw_classifier_conf,))
      (edw_train_correct, edw_test_correct) = parse_result(edw_output)

      print '- Running Weka classifier with configuration "%s"...' % (
             weka_classifier_conf,)
      weka_output = (
        weka('weka.classifiers.meta.AdaBoostM1 -t %s -T %s -W %s' % (
             train_disc_path, test_disc_path,
             weka_classifier_conf,)))
      assert not 'No boosting possible' in weka_output
      (weka_train_correct, weka_test_correct) = parse_result(weka_output)

      scores['edw_train'].append(edw_train_correct)
      scores['edw_test'].append(edw_test_correct)
      scores['weka_train'].append(weka_train_correct)
      scores['weka_test'].append(weka_test_correct)

      print 'training[e/w]:', edw_train_correct, weka_train_correct
      print 'testing[e/w]: ', edw_test_correct, weka_test_correct
      print

for ((ds_name, edw_c, weka_c,), scores) in all_scores:
  print ds_name
  print edw_c
  print weka_c
  print 'edw_train  ', scores['edw_train']
  print 'edw_test   ', scores['edw_test']
  print 'weka_train ', scores['weka_train']
  print 'weka_test  ', scores['weka_test']

