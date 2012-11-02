import typeAliases._

sealed abstract class DecisionTreeClassifier extends Classifier {
  def classify(xs: AttributeValueSeq): Option[Class]
}
case class Unknown() extends DecisionTreeClassifier {
  def classify(xs: AttributeValueSeq) = None
}
case class Branch(
  feature: Attribute, 
  choices: Map[AttributeValue, DecisionTreeClassifier]
) extends DecisionTreeClassifier {
  def classify(xs: AttributeValueSeq) = choices.get(xs(feature)) match {
    case Some(classifier) => classifier.classify(xs)
    case None => Some(1) // TODO FIXME
  }
}
case class Leaf(_class: Class) extends DecisionTreeClassifier {
  def classify(xs: AttributeValueSeq) = Some(_class)
}


object DecisionTreeClassifier {
  def chooseAttribute(features: Set[Attribute], instances: Seq[Instance]): Attribute = {
    def information(p: Seq[Instance]): Double = {
      val res = - p.groupBy(_._class).values.map { s_i =>
        val p_s_i = s_i.size/p.size.toDouble
        //jval k = (s_i.map(_.weight).sum/p.map(_.weight).sum) * p_s_i * math.log(p_s_i)/math.log(2)
        val k = p_s_i * math.log(p_s_i)/math.log(2)
        k
      }.sum
      res
    }

    def R(f: Attribute): Double = {
      val partitions = instances.groupBy(_.attributes(f)).values.toList
      partitions.map { p =>
        //(p.map(_.weight).sum / instances.map(_.weight).sum) * information(p)
        p.size / instances.size.toDouble * information(p)
      }.sum
    }

    def gain(f: Attribute): Double =
      information(instances) - R(f)
    
    //for (f <- features)
     // println(f + " " + gain(f))
    //println()

    features.minBy(R)
    //new scala.util.Random().shuffle(features.toSeq).minBy(R)
  }

  def buildFor(dataset: Dataset, 
               maxDepth: Int) = {
    def qdt(features: Set[Attribute], 
            instances: Seq[Instance], 
            depth: Int,
            default: Class = dataset.toSeq(0)._class): DecisionTreeClassifier = {
      val classes = instances.map(_._class)
      val allInstancesAreSameClass = classes.toSet.size == 1
      val mostCommonClass = 
        if (classes.isEmpty) default 
        else classes.groupBy(x => x).maxBy(_._2.size)_1
     
      if (instances.isEmpty)
        Leaf(default)
      else if (allInstancesAreSameClass)
        Leaf(instances(0)._class)
      else if (depth == maxDepth || features.isEmpty) 
        Leaf(mostCommonClass)
      else {
        val bestAttribute = chooseAttribute(features, instances)
        val classes = instances.map(_._class)
        Branch(bestAttribute, dataset.featureOptions(bestAttribute).map { 
          case v =>
            v -> qdt(features-bestAttribute, 
                     instances.filter(_.attributes(bestAttribute) == v), 
                     depth+1, 
                     mostCommonClass)
        }.toMap)
      }
    }

    qdt(dataset.features.toSet, dataset.instances, 0)
  }

  def describe(c: DecisionTreeClassifier, indent: String = ""): String = 
    c match {
      case Unknown() => indent + Red("RETURN UNKNOWN\n")
      case Branch(feature, choices) => (
          indent + Blue("SPLIT ON FEATURE ") + feature + "\n"
        + choices.map { case (v, cc) => (
              indent + Gray("|") + Yellow(" IF " + v) + Gray(" THEN\n")
            + describe(cc, indent + Gray("|   "))
          )}.mkString)
      case Leaf(_class) => indent + Green("RETURN CLASS " + _class) + "\n"
    }
}
