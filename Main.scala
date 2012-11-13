import aliases._
import utils._

object Main {
  def main(args: Array[String]) {
    val argsRegex = """^(\S+)\s+(\d+\.?\d*)\s+(\S+)$""".r

    argsRegex.findFirstIn(args.mkString(" ")) match {
      case None =>
        printHelpAndQuit("Invalid command line string.")

      case Some(argsRegex(datafilePath, trainingSetRatioRaw,
                          learnerConfRaw)) => {
        val dataset: Dataset = try {
          Dataset.fromCSVFile(datafilePath)
        } catch {
          case ex => printHelpAndQuit(ex.getMessage)
        }

        val trainingSetRatio = trainingSetRatioRaw.toDouble
        requireOrQuit(0 < trainingSetRatio && trainingSetRatio <= 1.0,
                      "Training set ratio must be in <0, 1]")

        val learnerConf = learnerConfReader.parse(learnerConfRaw)

        val isSingleClassifierRun = learnerConf.size == 1
        val learner =
          if (isSingleClassifierRun)
            learnerConf.head
          else
            ((ds: Dataset) =>
              new AdaBoostClassifier(learnerConf)(ds))

        val experiment = new Experiment(learner, dataset, trainingSetRatio)
        experiment.run()
      }
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
  ./analyse [file] [training set ratio] [(N*)classifier([opt...])...]

""" + Yellow("EXAMPLE:") + """
  ./analyse foo.cls 0.8 5*naivebayesian,2*decisiontree(maxDepth=A)

  Analyse the data set given in foo.cls, using 80% of the set for training.
  Use AdaBoost with five Naive Bayesian Classifiers, and two decision trees
  with maximum depth equal to the number of attrs in foo.cls (A)
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
    "naivebayesian" -> learnerReader(){ _ =>
      (dataset: Dataset) => new NaiveBayesianClassifier(dataset)
    },

    "decisiontree" -> learnerReader(
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
