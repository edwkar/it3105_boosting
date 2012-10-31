abstract class AnsiColor(val code: String) {
  def apply(x: Any) = code + x + "\033[1;m"
}
case object Gray extends AnsiColor("\033[1;30m")
case object Red extends AnsiColor("\033[1;31m")
case object Green extends AnsiColor("\033[1;32m")
case object Yellow extends AnsiColor("\033[1;33m")
case object Blue extends AnsiColor("\033[1;34m")
case object Magenta extends AnsiColor("\033[1;35m")
case object White extends AnsiColor("\033[1;37m")


object typeAliases {
  type Feature = Int
  type FeatureValue = Int
  type FeatureValueSeq = IndexedSeq[Feature]
  type Class = Int
  type Member = (FeatureValueSeq, Class)
}
import typeAliases._

class Dataset(val instances: IndexedSeq[Member]) extends Traversable[Member] {
  require(instances.isEmpty || Set(instances.map(_._1.size) : _*).size == 1, 
          "All members must have the same number of variables.")

  val features: Seq[Int] = (0 until instances(0)._1.size)

  val featureOptions: Map[Feature, Set[FeatureValue]] = (
    for (f <- features)
      yield f -> Set(instances.map(_._1(f)) : _*)
  ).toMap

  override def foreach[U](f: Member => U): Unit = instances.foreach(f)
}
object Dataset {
  def fromCSVFile(translatorFor: Traversable[String] => (String => Int))(path: String) = {
    val membersRaw = for {
      line <- Vector(scala.io.Source.fromFile(path).mkString.split("\n") : _*)
      xs = Vector(line.trim.split(",") : _*)
      (vars, _class) = (xs.dropRight(1), xs.last)
    } yield (vars, _class)

    val numVariables = membersRaw(0)._1.size
    val translators = (0 until numVariables).map{ v => 
      val values = membersRaw.map(_._1(v))
      translatorFor(values)
    }

    new Dataset(
      for ((vars, _class) <- membersRaw) 
        yield (vars.zipWithIndex.map { case (v, i) => translators(i)(v) },
               _class.toInt)
    )
  }
}

abstract trait AbstractClassifier {
  def classify(xs: FeatureValueSeq): Option[Class]

  def performanceOn(ds: Dataset) = {
    val numCorrect = ds.count(x => c.classify(x._1) == Some(x._2))
    (numCorrect, ds.size, numCorrect/ds.size.toDouble)
  }
}

sealed abstract class Classifier extends AbstractClassifier {
  def classify(xs: FeatureValueSeq): Option[Class]
}
case class Unknown() extends Classifier {
  def classify(xs: FeatureValueSeq) = None
}
case class Branch(feature: Feature, choices: Map[FeatureValue, Classifier]) 
extends Classifier {
  def classify(xs: FeatureValueSeq) = choices.get(xs(feature)) match {
    case Some(classifier) => classifier.classify(xs)
    case None => None
  }
}
case class Leaf(_class: Class) extends Classifier {
  def classify(xs: FeatureValueSeq) = Some(_class)
}

def entropy(p: Seq[Any]): Double =
  - p.groupBy(x => x).values.map { s_i => 
    val p_s_i = s_i.size/p.size.toDouble
    p_s_i * math.log(p_s_i)/math.log(2)
  }.sum

object DecisionTreeClassifier {
  def buildFor(dataset: Dataset, 
               chooseFeature: (Set[Feature], Seq[Member]) => Feature, 
               maxDepth: Int = 1200) = {

    def qdt(features: Set[Feature], 
            instances: Seq[Member], 
            depth: Int): Classifier = 
      if (entropy(instances) == 0.0) 
        Leaf(instances(0)._2)
      else if (depth == maxDepth+1 || (features.isEmpty && !instances.isEmpty)) 
        Unknown()
      else {
        val bestFeature = chooseFeature(features, instances)
        val partitions = instances.groupBy(_._1(bestFeature))
        Branch(bestFeature, partitions.map { case (v, s) =>
          v -> qdt(features-bestFeature, s, depth+1)
        }.toMap)
      }

    qdt(dataset.features.toSet, dataset.instances, 0)
  }

  def describe(c: Classifier, indent: String = ""): String = 
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

def chooseFeature(features: Set[Feature], instances: Seq[Member]): Feature = 
  features.maxBy(f => {
    val partitions = instances.groupBy(_._1(f)).values
    partitions.map { p => 
      (p.size / instances.size.toDouble) * entropy(p)
    }.sum
  })


def translatorFor(xs: Traversable[String]): (String => Int) = {
  val asDoubles = xs.map(_.toDouble)
  val areIntegers = asDoubles.forall(x => x == math.floor(x))
  if (areIntegers) 
    (x: String) => x.toInt
  else {
    val (minVal, maxVal) = (asDoubles.min, asDoubles.max)
    val numBins = asDoubles.toSet.size
    val binWidth = (maxVal-minVal)/numBins
    val bins = for (i <- 0 until numBins) 
                 yield ((minVal+i*binWidth, minVal+(i+1)*binWidth), i+1)

    (_x: String) => {
      val x = _x.toDouble
      bins.find(b => b._1._1 <= x && x <= b._1._2) match { 
        case Some(matchingBin) => matchingBin._2 
        case None if math.abs(maxVal-x) <= 1e-6  => bins.last._2
        case _ => throw new Exception("Could not map " + x + " to bin!")
      }
    }
  }
}

val ds = Dataset.fromCSVFile(translatorFor)("./datasets/yeast.txt")
val c = DecisionTreeClassifier.buildFor(ds, chooseFeature, 24)
println(DecisionTreeClassifier.describe(c))
println(c.performanceOn(ds))

