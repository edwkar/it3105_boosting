import typeAliases._

sealed abstract class DecisionTreeClassifier extends Classifier {
  def classify(xs: AttributeValueSeq): Option[Class]
}
case class Unknown() extends DecisionTreeClassifier {
  def classify(xs: AttributeValueSeq) = None
}
case class Branch(
  attribute: Attribute, 
  choices: Map[AttributeValue, DecisionTreeClassifier]
) extends DecisionTreeClassifier {
  def classify(xs: AttributeValueSeq) = choices.get(xs(attribute)) match {
    case Some(classifier) => classifier.classify(xs)
    case None => Some(1) // TODO FIXME
  }
}
case class Leaf(_class: Class) extends DecisionTreeClassifier {
  def classify(xs: AttributeValueSeq) = Some(_class)
}


object DecisionTreeClassifier {
  def chooseAttribute(
    attributes: Set[Attribute], 
    instances: Seq[Instance],
    parallelize: Boolean
  ): Attribute = {
    val ds = instances

    def gainRatio(a: Attribute) = {
      gain(a) / splitInfo(a)(ds)
    }

    def gain(a: Attribute) = 
      info(ds) - infoSplit(ds, a)

    def infoSplit(xs: Seq[Instance], a: Attribute) =
      xs.groupBy(_.attributes(a)).values.map {
        case ys: Seq[Instance] =>
          val v = ys.map(_.weight).sum / xs.map(_.weight).sum
          v * info(ys)
      }.sum

    def info(xs: Seq[Instance]) = {
      val res = - xs.groupBy(_._class).values.map { 
        case ys: Seq[Instance] => 
          val v = ys.map(_.weight).sum / xs.map(_.weight).sum
          v * math.log(v)/math.log(2)
      }.sum
      res
    }

    def splitInfo(a: Attribute)(xs: Seq[Instance]) = {
      val res = - xs.groupBy(_.attributes(a)).values.map {
        case ys: Seq[Instance] => 
          assert { ys.map(_.attributes(a)).toSet.size == 1 }
          val v = ys.map(_.weight).sum / xs.map(_.weight).sum
          v * math.log(v)/math.log(2)
      }.sum
      //println("si, " + a + "  = " + res)
      res
    }

    val bestAttribute = 
      if (parallelize || true)
        attributes.par.map((a: Attribute) => (a, gainRatio(a))).maxBy(_._2)._1
      else  
        attributes.map((a: Attribute) => (a, gainRatio(a))).maxBy(_._2)._1
    bestAttribute
  }

  def buildFor(dataset: Dataset, 
               maxDepth: Int) = {
    def qdt(attributes: Set[Attribute], 
            instances: Seq[Instance], 
            depth: Int,
            default: Class = 
              dataset.classes.maxBy(c => dataset.instances.filter(_._class == c).map(_.weight).sum)
    ): DecisionTreeClassifier = {
      val classes = instances.map(_._class)
      val allInstancesAreSameClass = classes.toSet.size == 1
      val mostCommonClass = 
        if (classes.isEmpty) 
          default 
        else 
          dataset.classes.maxBy(c => instances.filter(_._class == c).map(_.weight).sum)
     
      if (instances.isEmpty)
        Leaf(default)
      else if (allInstancesAreSameClass)
        Leaf(instances(0)._class)
      else if (depth == maxDepth || attributes.isEmpty) 
        Leaf(mostCommonClass)
      else {
        val bestAttribute = chooseAttribute(attributes, instances, depth <= 1)
        val classes = instances.map(_._class)
        Branch(bestAttribute, dataset.attributeOptions(bestAttribute).par.map { 
          case v =>
            v -> qdt(attributes-bestAttribute, 
                     instances.filter(_.attributes(bestAttribute) == v), 
                     depth+1, 
                     mostCommonClass)
        }.seq.toMap)
      }
    }

    qdt(dataset.attributes.toSet, dataset.instances, 0)
  }

  def describe(c: DecisionTreeClassifier, indent: String = ""): String = 
    c match {
      case Unknown() => indent + Red("RETURN UNKNOWN\n")
      case Branch(attribute, choices) => (
          indent + Blue("SPLIT ON FEATURE ") + attribute + "\n"
        + choices.map { case (v, cc) => (
              indent + Gray("|") + Yellow(" IF " + v) + Gray(" THEN\n")
            + describe(cc, indent + Gray("|   "))
          )}.mkString)
      case Leaf(_class) => indent + Green("RETURN CLASS " + _class) + "\n"
    }
}
