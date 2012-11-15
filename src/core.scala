object aliases {
  type Attr = Int
  type AttrValue = Int
  type AttrValueSeq = Seq[Attr]
  type Class = Int
  type Learner = Dataset => Classifier
}

import aliases._
import utils._


trait Classifier {
  def apply(xs: AttrValueSeq): Class

  def performanceOn(ds: Dataset) = {
    val numCorrect = ds.count(x => this(x.attrs) == x.cls)
    (numCorrect, ds.size, numCorrect/ds.size.toDouble)
  }
}


case class Instance(attrs: AttrValueSeq, cls: Class, weight: Double)


class Dataset(val instances: Seq[Instance]) extends Traversable[Instance] {
  require(allAreEqual(instances.map(_.attrs.size)),
          "All members must have the same number of variables.")
  require(math.abs(instances.map(_.weight).sum-1.0) <= 1e-4,
          "Weights must sum to 1. Current weight sum is: " + instances.map(_.weight).sum)

  val attrs = (0 until (instances(0).attrs.size)).toList
  val classes = instances.map(_.cls).toSet

  val attrOptions: Map[Attr, Set[AttrValue]] = (
    for (f <- attrs)
      yield f -> Set(instances.map(_.attrs(f)) : _*)
  ).toMap

  def shuffle = new Dataset(Dataset.R.shuffle(instances))

  def split(n: Int): (Dataset, Dataset) = {
    require(n <= size)
    val as = Dataset.normalizeWeights(instances.take(n))
    val bs = Dataset.normalizeWeights(instances.drop(n))
    (new Dataset(as), new Dataset(bs))
  }

  override def foreach[U](f: Instance => U): Unit = instances.foreach(f)
}


object Dataset {
  private val R = new scala.util.Random()

  def normalizeWeights(xs: Seq[Instance]): Seq[Instance] = {
    val wsum = xs.map(_.weight).sum
    xs.map { case Instance(a, c, w) => Instance(a, c, w/wsum) }
  }

  def fromCSVFile(path: String): Dataset = {
    val instancesRaw = Vector(scala.io.Source.fromFile(path).mkString.split("\n") : _*)
    new Dataset(
      for {
        instRaw <- instancesRaw
        xs = Vector(instRaw.trim.split(",") : _*).map(_.toInt)
        (attrs, cls) = (xs.dropRight(1), xs.last)
      } yield Instance(attrs, cls, 1.0/instancesRaw.size)
    )
  }
}
