package adaboost;

import aliases._

class NaiveBayesianClassifier(dataset: Dataset) extends Classifier {
type Key = (Class, Attr, AttrValue)
  type Probability = Double

  val minWeight = 0.01 * 1.0/dataset.size //1e-5 //math.max(1e-10, dataset.map(_.weight).min)

  override def apply(xs: AttrValueSeq) = {
    def calc(attrs: List[Attr], scores: Map[Class, Probability]): Class =
      attrs match {
        case Nil => scores.maxBy(_._2)._1
        case (a::as) =>
          val newScores = scores.map {
            case (c, p) => {
              assert(p != 0)
              assert(p_v_a_c(c, a, xs(a)) != 0)
              assert(p * p_v_a_c(c, a, xs(a)) != 0)
              c -> p * p_v_a_c(c, a, xs(a))
            }
          }
          val normalizedScores =
            if (newScores.exists(_._2 < 1e-75))
              newScores.mapValues { _ * 1e75 }
            else
              newScores

          if (normalizedScores.exists(_._2 == 0.0))
            throw new Exception("Numerical underflow in NBC calculation!")
          if (normalizedScores.exists(_._2.isInfinite))
            throw new Exception("Numerical overflow in NBC calculation!")
          calc(as, normalizedScores)
      }

    calc(dataset.attrs,
         dataset.classes.map(c => c -> math.max(minWeight, p_cls(c))).toMap)
  }

  private val p_cls: Map[Class, Probability] =
    (for (c <- dataset.classes)
      //yield c -> dataset.filter(_.cls == c).size * 1.0 // WTF?
      yield c -> dataset.filter(_.cls == c).map(_.weight).sum // WTH?
    ).toMap

  private val p_v_a_c: Map[Key, Probability] =
    (for {
      (cls, classInstances) <- dataset.groupBy(_.cls)
      attr <- dataset.attrs.par
      value <- dataset.attrOptions(attr)
    } yield {
      val xs = classInstances.filter(_.attrs(attr) == value).toList
      val key = (cls, attr, value)
      val p = xs.map(_.weight).sum /
                classInstances.toList.map(_.weight).sum
      key -> (if (p == 0.0) minWeight else p)
    }).seq.toMap
}
