package adaboost;

object utils {
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

  def allAreEqual[T](xs: Seq[T]) = xs.toSet.size <= 1
}
