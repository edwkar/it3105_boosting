import aliases._

class NaiveBayesianClassifier(dataset: Dataset) extends Classifier {
  type Key = (Class, Attr, AttrValue)
  type Probability = Double

  val minWeight = dataset.map(_.weight).min

  override def apply(xs: AttrValueSeq) =
    dataset.classes.par.maxBy { cls =>
      dataset.attrs.map { attr =>
        p.get( (cls, attr, xs(attr)) ) match {
          case Some(pp) => pp
          case None => minWeight
        }
      }.sorted.foldLeft(1.0)(_ * _) * p_cls(cls)
    }

  private val p_cls: Map[Class, Probability] =
    (for (c <- dataset.classes)
      yield c -> dataset.filter(_.cls == c).size * 1.0 // WTF?
      //yield c -> dataset.filter(_.cls == c).map(_.weight).sum // WTH?
    ).toMap

  private val p: Map[Key, Probability] =
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
