import typeAliases._

class NaiveBayesianClassifier(dataset: Dataset) extends Classifier {
  type Key = (Class, Attribute, AttributeValue)
  type Probability = Double

  def classify(xs: AttributeValueSeq) = 
    Some(dataset.classes.maxBy { _class => 
      dataset.features.map { feature => 
        p.get( (_class, feature, xs(feature)) ) match {
          case Some(pp) => pp
          case None => 1.0
        }
      }.foldLeft(p_class(_class))(_ * _)
    })
  
  private val p_class: Map[Class, Probability] = 
    (for (c <- dataset.classes) 
      yield c -> dataset.filter(_._class == c).map(_.weight).sum
    ).toMap


  private val p: Map[Key, Probability] = 
    (for {
      _class <- dataset.classes
      classInstances = dataset.filter(_._class == _class)
      feature <- dataset.features
      value <- dataset.featureOptions(feature)
    } yield {
      val xs = classInstances.filter(_.attributes(feature) == value)
      val key = (_class, feature, value)
      //key -> xs.map(_.weight).sum / classInstances.map(_.weight).sum
      key -> xs.size / classInstances.size.toDouble
    }).toMap
}
