import typeAliases._

class NaiveBayesianClassifier(dataset: Dataset) extends Classifier {
  type Key = (Class, Attribute, AttributeValue)
  type Probability = Double

  def classify(xs: AttributeValueSeq) = 
    Some(dataset.classes.par.maxBy { _class => 
      dataset.attributes.map { attribute => 
        p.get( (_class, attribute, xs(attribute)) ) match {
          case Some(pp) => pp
          case None => 1e-70
        }
      }.foldLeft(p_class(_class))(_ * _)
    })
  
  private val p_class: Map[Class, Probability] = 
    (for (c <- dataset.classes) 
      yield c -> dataset.par.filter(_._class == c).map(_.weight).sum
    ).toMap


  private val p: Map[Key, Probability] = 
    (for {
      (_class, classInstances) <- dataset.par.groupBy(_._class)
      attribute <- dataset.attributes.par
      value <- dataset.attributeOptions(attribute)
    } yield {
      val xs = classInstances.filter(_.attributes(attribute) == value)
      val key = (_class, attribute, value)
      key -> xs.map(_.weight).sum / classInstances.map(_.weight).sum
      //key -> xs.size / classInstances.size.toDouble
    }).seq.toMap
}
