import typeAliases._

class StumpClassifier(ds: Dataset) extends Classifier {
  def classify(xs: AttributeValueSeq) = {
    prediction.get(xs(bestAttribute)) match {
      case x @ Some(c) => x
      case _ => { println("n'est pas"); None }
    }
  }

  println("pre")
  val bestAttribute = ds.attributes.maxBy(gainRatio)
  val prediction = ds.attributeOptions(bestAttribute).map { opt =>
    val matching = ds.filter(_.attributes(bestAttribute) == opt)
    val answer = ds.classes.maxBy(c => matching.filter(_._class == c).map(_.weight).sum)
    opt -> answer
  }.toMap
  println(prediction)

  println("bestAttribute = " + bestAttribute + "  gainRatio = " + gainRatio(bestAttribute))

  def gainRatio(a: Attribute) = {
    gain(a) / splitInfo(a)(ds.instances)
  }

  def gain(a: Attribute) = 
    info(ds.instances) - infoSplit(ds.instances, a)

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
}
