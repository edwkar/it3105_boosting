object Main {
  def main(args: Array[String]) {


    println("""
Usage: 
  ./analyse [file] [training set ratio] [(N*)classifier([opt...])...] [output]

Example:
  ./analyse foo.cls 80% 5*naivebayesian,2*decisiontree(depth=A) console

  Analyse the data set given in foo.cls, using 80% of the set as training
  input.  Use AdaBoost with five Naive Bayesian Classifiers, and two decision
  trees with maximum depth equal to the number of attributes in foo.cls (A)
  Finally, yield beautified console output.
  """)
  }
}
