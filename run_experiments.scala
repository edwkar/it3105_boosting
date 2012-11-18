import scala.sys.process._

val RAW_DATASETS_PATH = "./datasets/raw"
val WEKA_PATH = "./weka.jar"
val TRAIN_SPLIT_PERCENT = 67
val DANCE_HALL = "/tmp"

val R = new scala.util.Random(42)

type ClassifierCount = Int
type TrainingSetPath = String
type TestingSetPath = String
case class Result(trainingAccuracy: Double,
                  testingAccuracy: Double) {
  require(0 <= trainingAccuracy && trainingAccuracy <= 1)
  require(0 <= testingAccuracy && testingAccuracy <= 1)
}

case class ComposedResult(results: Seq[Result]) {
  val meanTrainingAcc = results.map(_.trainingAccuracy).sum / results.size.toDouble
  val meanTestingAcc = results.map(_.testingAccuracy).sum / results.size.toDouble
}



def main() {
  runBoostingComparisionExperiments()
  //runBoostingPerformanceExperiments()
}
main()

def runBoostingComparisionExperiments() {
  def EA(conf: String): (PreparedDataset) => Result = null
  def WA(conf: String): (PreparedDataset) => Result = null

  val datasetPaths = List("yeast", "haberman", "nursery").map {x =>
                       RAW_DATASETS_PATH + "/" + x + ".arff"
                     }
  val classifierPairs = List(
    (EA("10dt"), WA("-I 10 weka.classifiers.trees.J48 -- -U")),
    (EA("10nb"), WA("-I 10 weka.classifiers.bayes.NaiveBayes"))
  )

  val roundsPer = 20

  for {
    datasetPath <- datasetPaths
    (ec, wk) <- classifierPairs
  } {
    for (r <- 0 until roundsPer) {
      val ds = new PreparedDataset(datasetPath, 67, R.nextInt(0x42424242))
    }
  }
}

def runBoostingPerformanceExperiments() {
  val f: (PreparedDataset, ClassifierCount) => Result = (ds, cc) => {
    val conf = "%ddt[maxDepth=3]".format(cc)
    val learner = adaboost.Main.parseLearnerConf(conf)
    val trainingSet = adaboost.Dataset.fromCSVFile(ds.trainPathDiscCsv)
    val testingSet = adaboost.Dataset.fromCSVFile(ds.testPathDiscCsv)

    val x@((trainingNumCorrect, _, _), (testingNumCorrect, _, _)) =
      adaboost.Main.trainAndTest(learner, trainingSet, testingSet)
    println(x)

    Result(trainingNumCorrect/ds.trainingSetSize.toDouble,
           testingNumCorrect/ds.testingSetSize.toDouble)
  }
  val performance = testBoostingPerformance(
    RAW_DATASETS_PATH + "/kr-vs-kp.arff",
    f,
    20
  )
  printToFile(new java.io.File("boosting_performance_res.m")){ pw =>
    pw.print("""
mean_training_acc_x = [%s];
mean_training_acc_y = [%s];
mean_testing_acc_x = [%s];
mean_testing_acc_y = [%s];
""".format(
      performance.map(_._1).mkString(","),
      performance.map(_._2.meanTrainingAcc).mkString(","),
      performance.map(_._1).mkString(","),
      performance.map(_._2.meanTestingAcc).mkString(",")))
  }
}

def testBoostingPerformance(
  datasetPath: String,
  f: (PreparedDataset, ClassifierCount) => Result,
  roundsPerCount: Int,
  classifierCounts: Seq[ClassifierCount] = List(1,2,3,4) ++ Range(5, 101, 5)
): Seq[(ClassifierCount, ComposedResult)] = {
  val sets = (0 until roundsPerCount).par.map { r =>
     new PreparedDataset(datasetPath,
                         TRAIN_SPLIT_PERCENT,
                         R.nextInt(0x42424242))
  }.seq
  classifierCounts.map { cc =>
    printf("testBoostingPerformance: Testing classifier count %d...\n", cc)
    cc ->
      ComposedResult(
        (for (r <- (0 until roundsPerCount)) yield {
          printf("testBoostingPerformance: Starting round %d out of %d for cc %d...\n",
                 r+1, roundsPerCount, cc)
          val ds = sets(r)
          f(ds, cc)
        }).seq
      )
  }.seq
}



class PreparedDataset(val dsPath: String, val splitPercent: Int, val seed: Int) {
  private val id = PreparedDataset.idCounter.incrementAndGet()
  def path(xs: String) = DANCE_HALL + "/.ds" + id + "." + xs
  val randPath = path("rand.arff")
  val trainPath = path("train.arff")
  val testPath = path("test.arff")

  val trainPathDiscArff = path("train.disc.arff")
  val testPathDiscArff = path("test.disc.arff")
  val trainPathDiscCsv = path("train.disc.csv")
  val testPathDiscCsv = path("test.disc.csv")

  println("- Randomizing...")
  printToFile(new java.io.File(randPath)) { pw =>
    pw.write(weka("weka.core.Instances randomize %d %s".format(seed, dsPath)))
  }

  printf("- Splitting into training and test sets (at %d percent)...\n", splitPercent)
  weka(("weka.filters.unsupervised.instance.RemovePercentage -P %d " +
       "-i %s -o %s -V").format(TRAIN_SPLIT_PERCENT, randPath, trainPath))
  weka(("weka.filters.unsupervised.instance.RemovePercentage -P %d " +
        "-i %s -o %s").format(TRAIN_SPLIT_PERCENT, randPath, testPath))

  println("- Discretizing...")
  weka(("weka.filters.supervised.attribute.Discretize " +
        "-b -i %s -o %s -r %s -s %s -Y -c last").format(
        trainPath, trainPathDiscArff, testPath, testPathDiscArff))

  println("- Massaging sets to restricted .csv form...")
  "python utils_src/arff2csv.py %s %s %s %s".format(
    trainPathDiscArff, testPathDiscArff,
    trainPathDiscCsv, testPathDiscCsv).!

  val trainingSetSize = ("wc -l "+trainPathDiscCsv).!!.split(" ")(0).toInt
  val testingSetSize  = ("wc -l "+testPathDiscCsv).!!.split(" ")(0).toInt
}
object PreparedDataset {
  private val idCounter = new java.util.concurrent.atomic.AtomicInteger
}


def weka(cmd: String) =
  Process("java -Xmx4000M "+cmd, None, "CLASSPATH" -> WEKA_PATH).!!

def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
  val pw = new java.io.PrintWriter(f)
  try { op(pw) } finally { pw.close() }
}












