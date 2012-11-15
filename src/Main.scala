import aliases._
import utils._


object Main {
  def main(args: Array[String]) {
    val argsRegex = """^(\S+)\s+(\S+)\s+(\S+)$""".r

    argsRegex.findFirstIn(args.mkString(" ")) match {
      case None =>
        printHelpAndQuit("Invalid command line string.")

      case Some(argsRegex(trainingSetPath,
                          testingSetPath,
                          learnerConfRaw)) => {
        val (trainingSet, testingSet) = try {
          import Dataset.{fromCSVFile => read}
          (read(trainingSetPath), read(testingSetPath))
        } catch {
          case ex => printHelpAndQuit(ex.getMessage)
        }

        val learnerConf = learnerConfReader.parse(learnerConfRaw)

        val isSingleClassifierRun = learnerConf.size == 1
        val learner = if (isSingleClassifierRun)
                        learnerConf.head
                      else
                        ((ds: Dataset) =>
                          new AdaBoostClassifier(learnerConf)(ds))

        trainAndTest(learner, trainingSet, testingSet)
      }
    }
  }

  def trainAndTest(learner: Learner, trainingSet: Dataset, testingSet: Dataset) {
    val classifier = learner(trainingSet)
    val trainingPerformance = classifier.performanceOn(trainingSet)
    val testingPerformance = classifier.performanceOn(testingSet)

    println(Yellow("TRAINING SET SIZE: ") + trainingSet.size + "\n")
    println(Blue("TRAINING:  ") + visualise(trainingPerformance))
    println(Blue("TEST    :  ") + visualise(testingPerformance))

    println("Error on training data")
    println("Correctly Classified Instances " + trainingPerformance._1)
    println("Error on test data")
    println("Correctly Classified Instances " + testingPerformance._1)
  }

  private def visualise(xs: (Int, Int, Double)) = xs match {
    case (numCorrect, numTotal, _) => {
      val split = (55*numCorrect/numTotal.toDouble).toInt
      (Gray("|") + Green("-")*split + Red("-")*(55-split) + Gray("|") +
      " " + "%-22s".format(numCorrect + "/" + Gray(numTotal)) +
            Yellow("%.2f%%".format(100.0*numCorrect/numTotal.toDouble)))
    }
  }

  def requireOrQuit(cond: => Boolean, errorMessage: => String) {
    if (!cond)
      printHelpAndQuit(errorMessage)
  }

  def printHelpAndQuit(errorMessage: String = null): Nothing = {
    if (errorMessage != null)
      println(Red("ERROR: ") + errorMessage + "\n")

    println("""
""" + Yellow("USAGE:") + """
  ./analyse [training file] [testing file] [(N*)classifier([opt...])...]

""" + Yellow("EXAMPLE:") + """
  ./analyse foo.train foo.test 5*nb,2*dt(maxDepth=A)
  """)

    System.exit(0)
    throw new Exception("Will never reach here.")
  }
}


object learnerConfReader {
  import Main.{requireOrQuit, printHelpAndQuit}

  def parse(rawConf: String): List[Learner] = {
    val hasComma = rawConf.indexOf(",") != -1
    val hasDigitFirst = rawConf(0).isDigit

    if (!hasComma && !hasDigitFirst)
      List(parseSingleLearner(rawConf))
    else if (hasComma)
      rawConf.split(",").toList.flatMap(parse)
    else { // hasDigitFirst
      val repRe = """^(\d+)(.*)$""".r
      repRe.findFirstIn(rawConf) match {
        case Some(repRe(numRepStr, conf)) =>
          parse(conf + (","+conf)*(numRepStr.toInt-1))
        case None => printHelpAndQuit("fu!" + rawConf); null
      }
    }
  }

  private def parseSingleLearner(rawConf: String): Learner = {
    val learnerRe = """^([^\[]+)(\[([^\]]+)\])?$""".r

    val trainerOpt: Option[Learner] =
      for {
        learnerRe(name, _, options) <- learnerRe.findFirstIn(rawConf)
        rawOptions = (
          for {
            opt <- (if (options != null) options.split("/").toList else List())
            kv = opt.split("=").toList
          } yield (kv(0) -> kv(1))
        ).toMap
        learnerReader <- LEARNER_READERS.get(name)
      }
        yield learnerReader(rawOptions)

    trainerOpt match {
      case Some(ct) => ct
      case None => printHelpAndQuit("bork!"); null
    }
  }

  private val LEARNER_READERS = List(
    "nb" -> learnerReader(){ _ =>
      (dataset: Dataset) => new NaiveBayesianClassifier(dataset)
    },

    "dt" -> learnerReader(
      "maxDepth" -> ((x: String) => x == "A" || x.forall(_.isDigit))
    ){ opts =>
      val maxDepth = opts.get("maxDepth") match {
        case Some("A") => -1
        case Some(n)   => n.toInt
        case None      => 0x42424242
      }

      (dataset: Dataset) =>
        DecisionTreeClassifier.trainFor(
          dataset,
          if (maxDepth == -1) dataset.attrs.size else maxDepth
        )
    }
  ).toMap

  private def learnerReader(
    optionsWithPredicates: (String, String => Boolean)*)(
    create: Map[String, String] => Learner
  ): Map[String, String] => Learner = {
    val optPredMap = optionsWithPredicates.toMap

    (optionsRaw: Map[String, String]) => {
      for (a <- optionsRaw.keys) {
        requireOrQuit(
          optPredMap.contains(a),
          "Unrecognized option: '" + a + "'"
        )

        requireOrQuit(
          optPredMap(a)(optionsRaw(a)),
          "'" + optionsRaw(a) + "' is an illegal value for option '" + a + "'"
        )
      }

      create(optionsRaw)
    }
  }
}
