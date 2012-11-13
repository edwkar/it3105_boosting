import aliases._

class NaiveBayesianClassifier(dataset: Dataset) extends Classifier {
  type Key = (Class, Attr, AttrValue)
  type Probability = Double

  override def apply(xs: AttrValueSeq) =
    dataset.classes.par.maxBy { cls =>
      dataset.attrs.map { attr =>
        p.get( (cls, attr, xs(attr)) ) match {
          case Some(pp) => pp
          case None => 1E-5
        }
      }.foldLeft(p_cls(cls))(_ * _)
    }

  private val p_cls: Map[Class, Probability] =
    (for (c <- dataset.classes)
      yield c -> dataset.par.filter(_.cls == c).map(_.weight).sum
    ).toMap


  private val p: Map[Key, Probability] =
    (for {
      (cls, classInstances) <- dataset.groupBy(_.cls)
      attr <- dataset.attrs.par
      value <- dataset.attrOptions(attr)
    } yield {
      val xs = classInstances.filter(_.attrs(attr) == value)
      val key = (cls, attr, value)
      key -> xs.map(_.weight).sum / classInstances.map(_.weight).sum
    }).seq.toMap
}
