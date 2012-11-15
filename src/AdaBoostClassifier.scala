import aliases._

class AdaBoostClassifier(learners: List[Learner])
                        (trainingSet: Dataset) extends Classifier {
  type AlphaVal = Double

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

  private def train(instances: Dataset, learners: List[Learner],
                    isFirst: Boolean = false): List[(Classifier, AlphaVal)] = {
    val K = instances.classes.size

    learners match {
      case Nil => Nil

      case (trainer :: remLearners) => {
        println(learners.size)

        val h = trainer(instances)

        val err = instances.map {
          case Instance(attrs, cls, weight) =>
            if (h(attrs) != cls)
              weight
            else
              0
        }.toList.sorted.sum

        val a = math.log((1-err)/(err+1e-14)) + math.log(K-1)

        val reWeighted = new Dataset(Dataset.normalizeWeights(
          instances.toSeq.map {
            case instance@Instance(attrs, cls, weight) =>
              val newWeight = weight * math.exp(a * (if (h(attrs) != cls) 1 else 0))
              assert(!newWeight.isNaN)
              Instance(attrs, cls, newWeight)
          }
        ))
        println(reWeighted.map(_.weight).sum)
        println("min = " + reWeighted.map(_.weight).min)
        println("max = " + reWeighted.map(_.weight).max)

        printf("error=%.2f alpha=%.2f...\n", err, a)
        if (!isFirst && (err == 0.0 || (1-err) <= 1.0/K-1E-3)) {
          println("Aborting training...")
          Nil //train(instances, remLearners)
        } else
          (h, a) :: train(reWeighted, remLearners)
      }
    }
  }
}
