object typeAliases {
  type Attribute = Int
  type AttributeValue = Int
  type AttributeValueSeq = Seq[Attribute]
  type Class = Int
  type ClassifierTrainer = Dataset => Classifier
}
import typeAliases._


case class Instance(attributes: AttributeValueSeq, _class: Class, weight: Double)


class Dataset(val instances: Seq[Instance]) extends Traversable[Instance] {
  require(instances.isEmpty || Set(instances.map(_.attributes.size) : _*).size == 1, 
          "All members must have the same number of variables.")
  require(math.abs(instances.map(_.weight).sum-1.0) <= 1e-4, 
          "Weights must sum to 1. Current weight sum is: " + instances.map(_.weight).sum)

  val features: Seq[Int] = (0 until instances(0).attributes.size)
  val classes: Set[Int] = instances.map(_._class).toSet

  val featureOptions: Map[Attribute, Set[AttributeValue]] = (
    for (f <- features)
      yield f -> Set(instances.map(_.attributes(f)) : _*)
  ).toMap

  def split(n: Int): (Dataset, Dataset) = {
    require(n <= size)
    val as = Dataset.normalizeWeights(instances.take(n))
    val bs = Dataset.normalizeWeights(instances.drop(n))
    (new Dataset(as), new Dataset(bs))
  }

  def shuffle = new Dataset(new scala.util.Random().shuffle(instances))

  override def foreach[U](f: Instance => U): Unit = instances.foreach(f)
}


object Dataset {
  type TranslatorFactory = Traversable[String] => (String => AttributeValue)

  def normalizeWeights(xs: Seq[Instance]): Seq[Instance] = {
    val wsum = xs.map(_.weight).sum
    xs.map { case Instance(a, c, w) => Instance(a, c, w/wsum) }
  }

  def fromCSVFile(path: String, 
                  translatorFor: TranslatorFactory = basicTranslatorFor) = {
    println("reading")
    val membersRaw = for {
      line <- Vector(scala.io.Source.fromFile(path).mkString.split("\n") : _*)
      xs = Vector(line.trim.split(",") : _*)
      (attributes, _class) = (xs.dropRight(1), xs.last)
    } yield (attributes, _class)
    println("done")

    val numVariables = membersRaw(0)._1.size
    val translators = (0 until numVariables).map{ v => 
      val attributes = membersRaw.map(_._1(v))
      translatorFor(attributes)
    }

    new Dataset(
      for ((attributes, _class) <- membersRaw) 
        yield Instance(attributes.zipWithIndex.map { 
                        case (v, i) => translators(i)(v) 
                       }, _class.toInt, 1.0/membersRaw.size)
    )
  }

  def basicTranslatorFor(xs: Traversable[String]) = {
    val asDoubles = xs.map(_.toDouble)
    val areIntegers = asDoubles.forall(x => x == math.floor(x))
    if (areIntegers) 
      (x: String) => x.toInt
    else {
      val (minVal, maxVal) = (asDoubles.min, asDoubles.max)
      val numBins = 5
      val binWidth = (maxVal-minVal)/numBins
      val bins = for (i <- 0 until numBins) 
                   yield ((minVal+i*binWidth, minVal+(i+1)*binWidth), i+1)

      (_x: String) => {
        val x = _x.toDouble
        bins.find(b => b._1._1 <= x && x <= b._1._2) match { 
          case Some(matchingBin) => { matchingBin._2 }
          case None if math.abs(maxVal-x) <= 1e-6  => bins.last._2
          case _ => throw new Exception("Could not map " + x + " to bin!")
        }
      }
    }
  }
}


trait Classifier {
  def classify(xs: AttributeValueSeq): Option[Class]

  def performanceOn(ds: Dataset) = {
    val numCorrect = ds.count(x => classify(x.attributes) == Some(x._class))
    (numCorrect, ds.size, numCorrect/ds.size.toDouble)
  }
}
