from subprocess import Popen, check_output, PIPE, STDOUT
import random
import re

random.seed(42)

SIMPLE_SETS = ['glass.arff', 'page-blocks.arff', 'yeast.arff',]
RAW_DATASETS_PATH = './datasets/raw'
WEKA_PATH = './weka.jar'
TRAIN_SPLIT_PERCENT = 67
DANCE_HALL = '/tmp'
BOOSTER_CLASS_FILES = './target/scala-2.9.1/classes'

def run(cmd_line, **extra_kw_args):
  return check_output('nice ' + cmd_line, shell=True, **extra_kw_args)

def weka(cmd):
  return run('java -Xmx12000M %s' % cmd, env={ 'CLASSPATH': WEKA_PATH })

def report_output(o):
  for line in o.split('\n'):
    print '  >', line

#print '(Re-)Compiling project...'
#report_output(run('sbt compile'))

ds_rand_path = DANCE_HALL + '/.ds.randomized.arff'
train_path = DANCE_HALL + '/.ds.train.arff'
test_path = DANCE_HALL + '/.ds.test.arff'
train_disc_path = DANCE_HALL + '/.ds.train.disc.arff'
test_disc_path = DANCE_HALL + '/.ds.test.disc.arff'
train_disc_csv_path = DANCE_HALL + '/.ds.train.disc.csv'
test_disc_csv_path = DANCE_HALL + '/.ds.test.disc.csv'

def parse_result(weka_output):
  m = re.match(
    ('^.*?Error on training data.*?Correctly Classified Instances\s+(\d+)\s' +
      '.*?Error on test data.*?Correctly Classified Instances\s+(\d+).*$'),
    weka_output, re.DOTALL | re.MULTILINE)
  #print weka_output
  return int(m.group(1)), int(m.group(2))

def run_experiment(ds_name, edw_classifier_conf, weka_classifier_conf,
                   num_reps):
  print 'Processing ds "%s"...' % ds_name

  scores = { 'edw_train':  [], 'edw_test':  [],
             'weka_train': [], 'weka_test': [] }

  for i in range(num_reps):
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
    run('python src/arff2csv.py %s %s %s %s' % (
        train_disc_path, test_disc_path,
        train_disc_csv_path, test_disc_csv_path,))
    train_set_size = int(run('wc -l '+train_disc_csv_path).split(' ')[0])
    test_set_size = int(run('wc -l '+test_disc_csv_path).split(' ')[0])


    print '- Running AdaBoost classifier with configuration "%s"...' % (
           edw_classifier_conf,)
    edw_output = run('scala -J-Xmx4000m -cp %s Main %s %s %s' % (
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
#        weka('weka.classifiers.bayes.NaiveBayes -t %s -T %s' % (
    #print weka_output

    #if 'No boosting possible' in weka_output:
    #  scores['edw_train'].append(None)
    #  scores['edw_test'].append(None)
    #  scores['weka_train'].append(None)
    #  scores['weka_test'].append(None)
    #  continue
    print 'TODO PUT BACK'
    (weka_train_correct, weka_test_correct) = parse_result(weka_output)

    scores['edw_train'].append(edw_train_correct/float(train_set_size))
    scores['edw_test'].append(edw_test_correct/float(test_set_size))
    scores['weka_train'].append(weka_train_correct/float(train_set_size))
    scores['weka_test'].append(weka_test_correct/float(test_set_size))

    print 'training[e/w]:', edw_train_correct, weka_train_correct, ' /', train_set_size
    print 'testing[e/w]: ', edw_test_correct, weka_test_correct, ' /', test_set_size
    print

  return scores



CLASSIFIER_CONFIGURATIONS = (
  ('1nb', 'weka.classifiers.trees.J48 -I 10 -- -U -O',),
  ('10nb', 'weka.classifiers.trees.J48 -I 10 -- -U -O',),
  #('200dt[maxDepth=1]', 'weka.classifiers.trees.J48 -I 200 -- -U -O',),
  #('20dt[maxDepth=5]', 'weka.classifiers.trees.J48 -I 20 -- -U -O',),
  #(#'80dt[maxDepth=A]', 'weka.classifiers.trees.J48 -I 80 -- -U -O',),
  #('20nb', 'weka.classifiers.bayes.NaiveBayes -I 20'),
)

output_file = open('o.m', 'w')
output_file.write('\n')
output_file.close()

all_scores = []
def main():
  num_reps = 10
  for ds_name in ('yeast',):# 'haberman', 'yeast', 'nursery',):
    for e_c, w_c in CLASSIFIER_CONFIGURATIONS:
      if ds_name == 'nursery' and ('20' in e_c or '80' in e_c):
        continue
      try:
        score_key = (ds_name, e_c, w_c, num_reps,)
        scores = run_experiment(ds_name, e_c, w_c, num_reps)
        all_scores.append((score_key, scores))

        output_file = open('o.m', 'a')
        filter_str = lambda x: ''.join(c for c in str(x))
        output_file.write("""
edw_train  = %s;
weka_train = %s;
edw_test   = %s;
weka_test  = %s;
figure('Position',[0,0,600,300]);
plot(edw_train, '-;EDW Training;', weka_train, '-;WEKA Training;',
     edw_test, '-;EDW Testing;',   weka_test, '-;WEKA Testing;', [0, 1], '*');
title('%s %s %s');
set(findall (gcf, '-property', 'linewidth'), 'linewidth', 2);
set(findall (gcf, '-property', 'markersize'), 'markersize', 12);
grid('on');
print -dpng '../public_html/plots/%s_%s_%s_%d.png';

      """ % (scores['edw_train'], scores['weka_train'],
             scores['edw_test'], scores['weka_test'],
             ds_name, e_c, w_c,
             ds_name, filter_str(e_c), filter_str(w_c), num_reps))
        output_file.close()
      except:
        print '~~~ EXPERIMENT FAILED ~~~'

main()


