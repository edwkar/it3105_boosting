import aliases._

class NaiveBayesianClassifier(dataset: Dataset) extends Classifier {
  type Key = (Class, Attr, AttrValue)
  type Probability = Double

  val minWeight = dataset.map(_.weight).min

  override def apply(xs: AttrValueSeq) = {
    def calc(attrs: List[Attr], scores: Map[Class, Probability]): Class =
      attrs match {
        case Nil => scores.maxBy(_._2)._1
        case (a::as) =>
          val newScores = scores.map {
            case (c, p) => c -> p * p_v_a_c(c, a, xs(a))
          }
          //printf("%.8f\n", normalizedScores.min)
          // Potential for numerical underflow.
          val normalizedScores =
            if (newScores.exists(_._2 < 1e-50))
              newScores.mapValues { _ * 1e50 }
            else
              newScores

          //printf("%.8f\n", normalizedScores.min)
          //if (normalizedScores.exists(_._2 == 0.0))
          //  throw new Exception("Numerical underflow in NBC calculation!")
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
      key -> xs.map(_.weight).sorted.sum /
                classInstances.toList.map(_.weight).sorted.sum
    }).seq.toMap
}
