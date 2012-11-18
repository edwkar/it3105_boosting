package adaboost;

import aliases._

class AdaBoostClassifier(learners: List[Learner])
                        (trainingSet: Dataset) extends Classifier {
  type AlphaVal = Double
  type Support = Double
  type ClassCount = Int

  /*
  override def apply(v: AttrValueSeq) = {
    val votes = hypotheses.par.map {
      case (h, a) => (h(v), a)
    }

    val bestAnswer = trainingSet.classes.par.maxBy {
      case x => (votes.collect {
        case (vote, a) if vote == x => a
      }.sum)
    }

    assert(bestAnswer == xapply(v))
    bestAnswer
  }
  */

  override def apply(v: AttrValueSeq) = {
    def collectVotes(weightedHypotheses: List[(Classifier, AlphaVal)],
                     cls2Support: Map[Class, Support]): Class =
      weightedHypotheses match {
        case Nil => cls2Support.maxBy(_._2)._1

        case ((h, a) :: remHypotheses) =>
          val vote = h(v)
          val newClass2Support = cls2Support + (vote -> (cls2Support(vote) + a))
          if (newClass2Support(vote) >= 0.5*maxAlphaSum)
            vote
          else
            collectVotes(remHypotheses, newClass2Support)
      }

    collectVotes(hypotheses, trainingSet.classes.map(c => (c -> 0.0)).toMap)
  }

  private val hypotheses =
    train(VARIANT_M1)(trainingSet, learners, true).sortBy(-_._2)
  private val maxAlphaSum = hypotheses.map(_._2).sum
  System.err.println(hypotheses.size)
  System.err.println(hypotheses.map(_._2).sum / hypotheses.size)
  System.err.println(hypotheses.map(_._2).max)
  System.err.println((hypotheses.map(_._2).max / ((hypotheses.map(_._2).sum / hypotheses.size))).toInt)

  private def train(variant: Variant)
                   (instances: Dataset,
                    learners: List[Learner],
                    isFirst: Boolean): List[(Classifier, AlphaVal)] = {
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

        val a = variant.alphaFormula(err, K)

        val reWeighted = new Dataset(Dataset.normalizeWeights(
          instances.toSeq.map {
            case instance@Instance(attrs, cls, weight) =>
              val newWeight = weight * math.exp(a * (if (h(attrs) != cls) 1 else 0))
              assert(!newWeight.isNaN)
              Instance(attrs, cls, newWeight)
          }
        ))
        println(reWeighted.map(_.weight).sum)
        System.err.println("  > min = " + reWeighted.map(_.weight).min)
        System.err.println("  > max = " + reWeighted.map(_.weight).max)
        System.err.println("  > error=" + err + "  alpha=" + a)

        if (!isFirst && (err == 0.0 || err >= variant.maximumError(K))) {
          System.err.println("Aborting training...")
          //throw new Exception()
          Nil //train(instances, remLearners)
        } else
          (h, a) :: train(variant)(reWeighted, remLearners, false)
      }
    }
  }


  class Variant(
    val alphaFormula: (Error, ClassCount) => AlphaVal,
    val maximumError: ClassCount => Error
  )

  lazy val VARIANT_M1 = new Variant(
    (err, K) => math.log((1-err)/(err+1e-14)),
    _ => 0.5
  )

  lazy val VARIANT_SAMME = new Variant(
    (err, K) => math.log((1-err)/(err+1e-14)) + math.log(K-1),
    K => 1.0 - 1.0/K
  )
}
