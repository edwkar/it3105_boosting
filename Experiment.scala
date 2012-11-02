import typeAliases._

class Experiment(ct: ClassifierTrainer, ds: Dataset, trainingSetRatio: Double) {
  def run() {
    val trainingSetSize = math.floor(ds.size * trainingSetRatio).toInt
    val (trainingSet, testingSet) = ds.shuffle.split(trainingSetSize)
    println("split...")

    val classifier = ct(trainingSet)
    classifier match {
      case dtc: DecisionTreeClassifier =>
        null //println(DecisionTreeClassifier.describe(dtc))
      case _ =>
    }

    println(Yellow("TRAINING SET SIZE: ") + trainingSetSize + "\n")
    println(Yellow("CLASSIFIER"))
    println(Blue("TRAINING:  ") + visualise(classifier.performanceOn(trainingSet)))
    println(Blue("TEST    :  ") + visualise(classifier.performanceOn(testingSet)))
  }

  private def visualise(xs: (Int, Int, Double)) = xs match {
    case (numCorrect, numTotal, _) => {
      val split = (55*numCorrect/numTotal.toDouble).toInt
      (Gray("|") + Green("-")*split + Red("-")*(55-split) + Gray("|") + 
      " " + "%-22s".format(numCorrect + "/" + Gray(numTotal)) + 
            Yellow("%.2f%%".format(100.0*numCorrect/numTotal.toDouble)))
    }
  }
}
