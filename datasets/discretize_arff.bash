INPUT_FILE=$1

export CLASSPATH=./weka.jar 

java weka.core.Instances randomize 42 $INPUT_FILE > randomized.arff
java weka.filters.unsupervised.instance.RemovePercentage -P 66 \
     -i randomized.arff -o a.arff 
java weka.filters.unsupervised.instance.RemovePercentage -P 66 \
     -i randomized.arff -o b.arff -V

COLS_TO_DISCRETIZE=$2
CLASS_COL=$3
  java -Xmx2000M weka.filters.supervised.attribute.Discretize\
       -b -i a.arff -o a_disc.arff -Y\
          -r b.arff -s b_disc.arff -R $COLS_TO_DISCRETIZE -c last

java weka.core.Instances append a_disc.arff b_disc.arff > disc.arff

sed -i -e "s/...B\([0-9]\+\)of\([0-9]\+\).../\1/g" disc.arff
sed -i -e "s/...All.../1/g" disc.arff

CLASSPATH=weka.jar java weka.classifiers.meta.AdaBoostM1 \
  -D -I 200 \
  -t disc.arff \
  -W weka.classifiers.bayes.NaiveBayes

  #disc.arff \
