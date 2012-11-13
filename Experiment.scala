import aliases._
import utils._

class Experiment(ct: Learner, ds: Dataset, trainingSetRatio: Double) {
  def run() {
    val trainingSetSize = math.floor(ds.size * trainingSetRatio).toInt
    val (trainingSet, testingSet) =
      if (trainingSetSize == ds.size)
        (ds, ds)
      else
        ds.shuffle.split(trainingSetSize)

    val classifier = ct(trainingSet)

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
