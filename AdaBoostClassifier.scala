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

  private def train(instances: Dataset, trainers: List[Learner],
                    isFirst: Boolean = false): List[(Classifier, Double)] = {
    val K = instances.classes.size

    trainers match {
      case Nil => Nil

      case (trainer :: remLearners) => {
        val classifier = trainer(instances)

        val err = math.max(1e-10, instances.map {
          case Instance(attrs, cls, weight) =>
            if (classifier(attrs) != cls)
              weight
            else
              0
        }.sum)

        val a = math.log((1-err)/err) + math.log(K-1)

        val reWeighted = new Dataset(Dataset.normalizeWeights(
          instances.toSeq.map {
            case instance@Instance(attrs, cls, weight) =>
              val newWeight = weight * math.exp(a * (if (classifier(attrs) != cls) 1 else 0))
              assert(!newWeight.isNaN)
              Instance(attrs, cls, newWeight)
          }
        ))

        printf("error=%.2f alpha=%.2f...\n", err, a)
        if (!isFirst && (1-err) < 1.0/K)
          Nil //train(instances, remLearners)
        else
          (classifier, a) :: train(reWeighted, remLearners)
      }
    }
  }
}
