import typeAliases._

object Main {
  def main(args: Array[String]) {
    val argsRegex = """^(\S+)\s+(\d+\.?\d*)\s+(\S+)$""".r
    argsRegex.findFirstIn(args.mkString(" ")) match {
      case None =>
        printHelpAndQuit("Invalid command line string.")

      case Some(argsRegex(datafilePath, trainingSetRatioRaw,
                          classifierConfigurationRaw)) =>
        val dataset: Dataset = try {
          Dataset.fromCSVFile(datafilePath)
        } catch { case ex => printHelpAndQuit(ex.getMessage); null }
        val trainingSetRatio = trainingSetRatioRaw.toDouble
        requireOrQuit(trainingSetRatio > 0 && trainingSetRatio <= 1.0,
                      "Training set ratio must be in <0, 1]")
        val trainerConfiguration = parseTrainers(classifierConfigurationRaw)
        val trainer = 
          if (trainerConfiguration.size == 1)
            trainerConfiguration.head
          else
            ((ds: Dataset) =>
              new AdaBoostClassifier(trainerConfiguration)(ds))
        
      new Experiment(trainer, dataset, trainingSetRatio).run()
    }
  }

  private def requireOrQuit(cond: => Boolean, errorMessage: => String) {
    if (!cond)
      printHelpAndQuit(errorMessage)
  }

  private def parseTrainers(configuration: String): List[ClassifierTrainer] = {
    val hasComma = configuration.indexOf(",") != -1  
    val hasDigitFirst = configuration(0).isDigit

    if (!hasComma && !hasDigitFirst) 
      List(parseSingleTrainer(configuration))
    else if (hasComma)
      configuration.split(",").toList.flatMap(parseTrainers)
    else { // hasDigitFirst
      val repRe = """^(\d+)(.*)$""".r
      repRe.findFirstIn(configuration) match {
        case Some(repRe(numRepStr, conf)) =>
          parseTrainers(conf + (","+conf)*(numRepStr.toInt-1))
        case None => printHelpAndQuit("fu!" + configuration); null  
      }
    }
  }

  private def parseSingleTrainer(configuration: String): ClassifierTrainer = {
    val trainerRe = """^([^\[]+)(\[([^\]]+)\])?$""".r
    val cto: Option[ClassifierTrainer] = 
      for {
        trainerRe(name, _, options) <- trainerRe findFirstIn configuration
        rawOptions = (
          for {
            or <- (if (options != null) options.split("/").toList else List())
            kv = or.split("=").toList
          } yield (kv(0) -> kv(1))
        ).toMap
        trainerBuilder <- CLASSIFIER_TRAINER_BUILDERS.get(name)
      } yield trainerBuilder(rawOptions)

    cto match {
      case Some(ct) => ct
      case None => printHelpAndQuit("bork!"); null
    }
  }

  private val CLASSIFIER_TRAINER_BUILDERS = List(
    "naivebayesian" -> classifierReader(){ _ =>
      (dataset: Dataset) => new NaiveBayesianClassifier(dataset)
    },

    "decisiontree" -> classifierReader(
      "maxDepth" -> ((x: String) => x == "A" || x.forall(_.isDigit))
    ){ opts => 
      val maxDepth = opts.get("maxDepth") match {
        case Some("A") => -1
        case Some(n)   => n.toInt
        case None      => 42424242
      }

      (dataset: Dataset) => 
        DecisionTreeClassifier.buildFor(
          dataset,
          if (maxDepth == -1) dataset.features.size else maxDepth
        )
    },

    "stump" -> classifierReader(){ _ =>
      (dataset: Dataset) => new StumpClassifier(dataset)
    }
  ).toMap

  private def classifierReader(
    optionsAndPredicates: (String, String => Boolean)*)(
    create: Map[String, String] => ClassifierTrainer
  ): Map[String, String] => ClassifierTrainer = {
    val optPredMap = optionsAndPredicates.toMap
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


  private def printHelpAndQuit(errorMessage: String = null) {
    if (errorMessage != null)
      println(Red("ERROR: ") + errorMessage + "\n")

    println("""
""" + Yellow("USAGE:") + """
  ./analyse [file] [training set ratio] [(N*)classifier([opt...])...] 

""" + Yellow("EXAMPLE:") + """
  ./analyse foo.cls 0.8 5*naivebayesian,2*decisiontree(maxDepth=A) 

  Analyse the data set given in foo.cls, using 80% of the set for training.
  Use AdaBoost with five Naive Bayesian Classifiers, and two decision trees
  with maximum depth equal to the number of attributes in foo.cls (A)
  """)

    System.exit(0)
  }
}
