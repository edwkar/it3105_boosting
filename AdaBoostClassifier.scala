import typeAliases._

class AdaBoostClassifier(classifierTrainers: List[ClassifierTrainer])
                        (trainingSet: Dataset) extends Classifier {

  override def classify(v: AttributeValueSeq) = {
    val votes = hypotheses.map {
      case (h, z) => (h.classify(v), z)
    }
    val bestAnswer = trainingSet.classes.maxBy {
      case c1 => (votes.collect {
        case (Some(c2), z) if c1 == c2 => z
      }.sum)
    }
    Some(bestAnswer)
    hypotheses(0)._1.classify(v)
  }

  private val hypotheses = build(trainingSet, classifierTrainers)

  private def build(_weightedInstances: Dataset, 
                    trainers: List[ClassifierTrainer]): List[(Classifier, Double)] = 
    trainers match {
      case Nil => Nil

      case (trainer :: remTrainers) => {
        val minWeight = _weightedInstances.map(_.weight).min
        val maxWeight = _weightedInstances.map(_.weight).max
        var scaleDown = 1.0
        while ((maxWeight/minWeight)/scaleDown >= 500) {
          println((maxWeight/minWeight)/scaleDown)
          scaleDown *= 1.5 
        }
        val weightedInstances = new Dataset(Dataset.normalizeWeights(for {
          inst <- _weightedInstances.instances
          cnt = 1 + math.round((inst.weight/minWeight)/scaleDown).toInt
          k <- 0 until cnt
        } yield inst).toSeq)


        val classifier = trainer(weightedInstances)
        //classifier match {
         // case dtc: DecisionTreeClassifier => println(DecisionTreeClassifier.describe(dtc))
        //}

        val error = _weightedInstances.map { 
          case Instance(attributes, _class, weight) =>
            if (classifier.classify(attributes) != Some(_class))
              weight
            else
              0
        }.sum
        
        println("error: " + error)
        println("error: " + error/(1-error))

        val reWeightedInstances = new Dataset(Dataset.normalizeWeights(
          _weightedInstances.toSeq.map { 
            case instance@Instance(attributes, _class, weight) =>
              if (classifier.classify(attributes) == Some(_class))
                Instance(attributes, _class, weight * error/(1-error))
              else
                instance
          }
        ))

        println("ERROR: " + error)
        val z = math.log((1-error)/error)
        if (error >= 0.5)
          Nil
        else  
          (classifier, z) :: build(reWeightedInstances, remTrainers)
      }
    }
}
