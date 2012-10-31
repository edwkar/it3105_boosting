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
  val classes: Set[Int] = instances.map(_._2).toSet

  val featureOptions: Map[Feature, Set[FeatureValue]] = (
    for (f <- features)
      yield f -> Set(instances.map(_._1(f)) : _*)
  ).toMap

  def split(n: Int): (Dataset, Dataset) = {
    require(n <= size)
    (new Dataset(instances.take(n)), new Dataset(instances.drop(n)))
  }

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
    val numCorrect = ds.count(x => classify(x._1) == Some(x._2))
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

def entropy(p: Seq[Member]): Double =
  - p.groupBy(_._2).values.map { s_i => 
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
      if (math.abs(entropy(instances)) < 1e-6) 
        Leaf(instances(0)._2)
      else if (depth == maxDepth+1 || (features.isEmpty && !instances.isEmpty)) {
        if (Set(instances.map(_._2) : _*).size == 1)
          Leaf(instances(0)._2)
        else {
          val classes = new util.Random().shuffle(instances.map(_._2))
          Leaf(classes.maxBy(c => classes.count(_ == c)))
        }
      } else {
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

class RandomClassifier extends AbstractClassifier {
  val r = new util.Random()
  def classify(xs: FeatureValueSeq) = Some(if (r.nextDouble() >= .5) 1 else 2)
}

class NaiveBayesianClassifier(dataset: Dataset) extends AbstractClassifier {
  type Key = (Class, Feature, FeatureValue)
  type Probability = Double

  def classify(xs: FeatureValueSeq) = 
    Some(dataset.classes.maxBy { _class => 
      dataset.features.map { feature => 
        p.get( (_class, feature, xs(feature)) ) match {
          case Some(pp) => pp
          case None => 0.0
        }
      }.foldLeft(p_class(_class))(_ * _)
    })
  
  private val p_class: Map[Class, Probability] = 
    (for (c <- dataset.classes) 
      yield c -> dataset.count(_._2 == c)/dataset.size.toDouble
    ).toMap


  private val p: Map[Key, Probability] = 
    (for {
      _class <- dataset.classes
      classInstances = dataset.filter(_._2 == _class)
      feature <- dataset.features
      value <- dataset.featureOptions(feature)
    } yield {
      val xs = classInstances.filter(_._1(feature) == value)
      val key = (_class, feature, value)
      key -> xs.size/classInstances.size.toDouble
    }).toMap
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
    val numBins = 8 //asDoubles.toSet.size
    val binWidth = (maxVal-minVal)/numBins
    val bins = for (i <- 0 until numBins) 
                 yield ((minVal+i*binWidth, minVal+(i+1)*binWidth), i+1)

    //val xs = asDoubles.toSet.toList.sorted
    //val bins = (xs.zip(xs.drop(1))).zip(1 to numBins) 

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


val r = new util.Random()
for (trainingSetSize <- List(210, 270)) {
  println(Gray("-"*40) + " " + Yellow(trainingSetSize) + " " + Gray("-"*40) + "\n")

  def visualise(xs: (Int, Int, Double)) = xs match {
    case (numCorrect, numTotal, _) => {
      val split = (55*numCorrect/numTotal.toDouble).toInt
      (Gray("|") + Green("-")*split + Red("-")*(55-split) + Gray("|") + 
       " " + numCorrect + "/" + Gray(numTotal))
    }
  }


  {
    val ds = new Dataset(r.shuffle(Dataset.fromCSVFile(translatorFor)("./datasets/haberman.txt").instances))
    val (trainingSet, testSet) = ds.split(trainingSetSize)
    val c0 = new NaiveBayesianClassifier(trainingSet)
    val dt2 = DecisionTreeClassifier.buildFor(trainingSet, chooseFeature, 2)
    val dt4 = DecisionTreeClassifier.buildFor(trainingSet, chooseFeature, 4)
    val dt6 = DecisionTreeClassifier.buildFor(trainingSet, chooseFeature, 6)
    val dt8 = DecisionTreeClassifier.buildFor(trainingSet, chooseFeature, 8)
    val dt99 = DecisionTreeClassifier.buildFor(trainingSet, chooseFeature)
    val R = new RandomClassifier()
    //println(DecisionTreeClassifier.describe(c))
    println(Blue("NBC, TRAINING:  ") + visualise(c0.performanceOn(trainingSet)))
    println(Blue("NBC, TEST:      ") + visualise(c0.performanceOn(testSet)))
    println(Magenta( "DT2, TRAINING:  ") + visualise(dt2.performanceOn(trainingSet)))
    println(Magenta( "DT4, TRAINING:  ") + visualise(dt4.performanceOn(trainingSet)))
    println(Magenta( "DT6, TRAINING:  ") + visualise(dt6.performanceOn(trainingSet)))
    println(Magenta( "DT8, TRAINING:  ") + visualise(dt8.performanceOn(trainingSet)))
    println(Magenta( "DT99,TRAINING:  ") + visualise(dt99.performanceOn(trainingSet)))
    println(Magenta( "DT2, TEST:      ") + visualise(dt2.performanceOn(testSet)))
    println(Magenta( "DT4, TEST:      ") + visualise(dt4.performanceOn(testSet)))
    println(Magenta( "DT6, TEST:      ") + visualise(dt6.performanceOn(testSet)))
    println(Magenta( "DT8, TEST:      ") + visualise(dt8.performanceOn(testSet)))
    println(Magenta( "DT99,TEST:      ") + visualise(dt99.performanceOn(testSet)))
    println(Magenta( "RANDOM,TRAINING:") + visualise(R.performanceOn(testSet)))
    println(Magenta( "RANDOM,TEST:    ") + visualise(R.performanceOn(testSet)))
    println()
  }


  /*
  {
    val ds = Dataset.fromCSVFile(translatorFor)("./datasets/yeast.txt")
    val (trainingSet, testSet) = ds.split(trainingSetSize)
    val c0 = new NaiveBayesianClassifier(trainingSet)
    val c1 = DecisionTreeClassifier.buildFor(trainingSet, chooseFeature)
    //println(DecisionTreeClassifier.describe(c))
    println(Blue("NBC: ") + c0.performanceOn(testSet))
    println(Red( "DT:  ") + c1.performanceOn(testSet))
    println(Blue("NBC: ") + c0.performanceOn(trainingSet))
    println(Red( "DT:  ") + c1.performanceOn(trainingSet))
  }
  println()
  */
}
