import aliases._

class AdaBoostClassifier(learners: List[Learner])
                        (trainingSet: Dataset) extends Classifier {

  override def apply(v: AttrValueSeq) = {
    val votes = hypotheses.par.map {
      case (h, a) => (h(v), a)
    }

    val bestAnswer = trainingSet.classes.par.maxBy {
      case x => (votes.collect {
        case (vote, a) if vote == x => a
      }.sum)
    }

    bestAnswer
  }

  private val hypotheses = train(trainingSet, learners, true)

  private def train(weightedInstances: Dataset,
                    trainers: List[Learner],
                    isFirstLearner: Boolean = false): List[(Classifier, Double)] = {
    val K = weightedInstances.classes.size
    trainers match {
      case Nil => Nil

      case (trainer :: remLearners) => {
        val classifier = trainer(weightedInstances)

        assert { math.abs(weightedInstances.instances.map(_.weight).sum-1.0) < 1e-6 }

        val error = math.max(1e-8, weightedInstances.map {
          case Instance(attrs, cls, weight) =>
            if (classifier(attrs) != cls)
              weight
            else
              0
        }.sum)


        val a = math.log((1-error) / error) + math.log(K-1)

        val reWeightedInstances = new Dataset(Dataset.normalizeWeights(
          weightedInstances.toSeq.map {
            case instance@Instance(attrs, cls, weight) =>
              val newWeight = weight * math.exp(a * (if (classifier(attrs) != cls) 1 else 0))
              Instance(attrs, cls, newWeight)
          }
        ))

        println("ERROR: " + error)

        if (!isFirstLearner && (1-error) < 1.0/K)
          Nil
        else {
          println("got it")
          (classifier, a) :: train(reWeightedInstances, remLearners)
        }
      }
    }
  }
}
