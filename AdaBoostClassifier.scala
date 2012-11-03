import typeAliases._

class AdaBoostClassifier(classifierTrainers: List[ClassifierTrainer])
                        (trainingSet: Dataset) extends Classifier {

  override def classify(v: AttributeValueSeq) = {
    val votes = hypotheses.map {
      case (h, z) => (h.classify(v), z)
    }
    val bestAnswer = (None :: (trainingSet.classes.map(c => Some(c)).toList)).maxBy {
      case x => (votes.collect {
        case (vote, z) if vote == x => z
      }.sum)
    }
    bestAnswer
  }

  private val hypotheses = build(trainingSet, classifierTrainers, true)

  private def build(weightedInstances: Dataset, 
                    trainers: List[ClassifierTrainer],
                    isFirstTrainer: Boolean = false): List[(Classifier, Double)] = 
    trainers match {
      case Nil => Nil

      case (trainer :: remTrainers) => {
        val classifier = trainer(weightedInstances)
        //println(classifier)

        assert { math.abs(weightedInstances.instances.map(_.weight).sum-1.0) < 1e-6 }

        val error = math.max(1e-8, weightedInstances.map { 
          case Instance(attributes, _class, weight) =>
            if (classifier.classify(attributes) != Some(_class))
              weight
            else
              0
        }.sum)
        
        val reWeightedInstances = new Dataset(Dataset.normalizeWeights(
          weightedInstances.toSeq.map { 
            case instance@Instance(attributes, _class, weight) =>
              if (classifier.classify(attributes) == Some(_class)) 
                Instance(attributes, _class, weight * error/(1-error))
              else
                instance
          }
        ))

        println("ERROR: " + error)
        val z = math.log((1-error+1e-6) / error)
        //val z = math.log(1.0/error) //TODO
        //println(z)
        
        if (!isFirstTrainer && (math.abs(error-0.5) <= 1e-8 || math.abs(error-0.0) <= 1e-8))
          Nil
        else {
          println("got it")
          (classifier, z) :: build(reWeightedInstances, remTrainers)
        }
      }
    }
}
