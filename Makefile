all: AdaBoost/target/scala-2.9.1/classes/adaboost/*.class
	scala -J-Xmx6000m -classpath .:AdaBoost/target/scala-2.9.1/classes/ run_experiments.scala

AdaBoost/target/scala-2.9.1/classes/adaboost/*.class: AdaBoost/src/adaboost/*.scala
	cd AdaBoost && sbt compile
