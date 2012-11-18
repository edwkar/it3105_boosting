package adaboost;

import aliases._
import utils._




object DecisionTreeClassifier {
  val R = new scala.util.Random()

  def trainFor(dataset: Dataset, maxDepth: Int) = {
    val mostCommonInWholeSet = dataset.classes.maxBy(c =>
                                 dataset.instances.filter(_.cls == c).map(_.weight).sum)

    def growTree(attrs: Set[Attr], instances: Seq[Instance], depth: Int,
                 default: Class = mostCommonInWholeSet): DTC = {
      val classes = instances.map(_.cls)
      val allSameClass = classes.toSet.size == 1
      val mostCommonClass =
        if (classes.isEmpty)
          default
        else
          dataset.classes.maxBy(c => instances.filter(_.cls == c).map(_.weight).sum)

      if (instances.isEmpty)
        Leaf(default)
      else if (allSameClass || depth == maxDepth || attrs.isEmpty)
        Leaf(mostCommonClass)
      else {
        val bestAttr = chooseAttr(attrs, instances)
        val classes = instances.map(_.cls)
        Branch(
          bestAttr,
          dataset.attrOptions(bestAttr).par.map {
            case v =>
              v -> growTree(attrs-bestAttr,
                            instances.filter(_.attrs(bestAttr) == v),
                            depth+1,
                            mostCommonClass)
          }.seq.toMap,
          mostCommonClass
        )
      }
    }

    growTree(dataset.attrs.toSet, dataset.instances, 0)
  }

  def chooseAttr(attrs: Set[Attr], instances: Seq[Instance]): Attr = {
    // TODO!
    def gainRatio(a: Attr) =
        gain(a) / math.max(1e-8, splitInfo(instances, a))

    def gain(a: Attr) = /*info(instances) -*/ - infoSplitted(instances, a)

    def info(xs: Seq[Instance]) =
      - xs.groupBy(_.cls).values.map {
        case ys: Seq[Instance] =>
          val v = ys.map(_.weight).sum / xs.map(_.weight).sum
          v * math.log(v)/math.log(2)
      }.sum

    def infoSplitted(xs: Seq[Instance], a: Attr) =
      xs.groupBy(_.attrs(a)).values.map {
        case ys: Seq[Instance] =>
          val v = ys.map(_.weight).sum / xs.map(_.weight).sum
          //val v = ys.size / xs.size.toDouble
          v * info(ys)
      }.sum

    def splitInfo(xs: Seq[Instance], a: Attr) =
      - xs.groupBy(_.attrs(a)).values.map {
        case ys: Seq[Instance] =>
          assert { ys.map(_.attrs(a)).toSet.size == 1 }
          //val v = ys.map(_.weight).sum / xs.map(_.weight).sum
          val v = ys.size / xs.size.toDouble
          v * math.log(v)/math.log(2)
      }.sum

    val bestAttr =
      R.shuffle(attrs.toList).par.map((a: Attr) => (a, gain(a))).maxBy(_._2)._1
      //attrs.par.map((a: Attr) => (a, gainRatio(a))).maxBy(_._2)._1
    bestAttr
  }


  sealed abstract class DTC extends Classifier

  case class Branch(
    attr: Attr,
    choices: Map[AttrValue, DTC],
    fallbackChoice: Class
  ) extends DTC {
    override def apply(xs: AttrValueSeq) = choices.get(xs(attr)) match {
      case Some(classifier) => classifier(xs)
      case None => fallbackChoice
    }
  }

  case class Leaf(cls: Class) extends DTC { def apply(xs: AttrValueSeq) = cls }


  def describe(c: DTC, indent: String = ""): String =
    c match {
      case Branch(attr, choices, fallbackChoice) => (
          indent + Blue("SPLIT ON FEATURE ") + attr + "\n"
        + choices.map { case (v, cc) => (
              indent + Gray("|") + Yellow(" IF " + v) + Gray(" THEN\n")
            + describe(cc, indent + Gray("|   "))
          )}.mkString)
      case Leaf(cls) => indent + Green("RETURN CLASS " + cls) + "\n"
    }
}
