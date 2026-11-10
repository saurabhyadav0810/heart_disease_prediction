// =============================================================================
// Heart Disease Prediction and Healthcare Analytics
// Build Configuration
// =============================================================================

name := "heart-disease-prediction"
version := "1.0.0"
scalaVersion := "2.12.20"

// Apache Spark 3.5.8 dependencies
val sparkVersion = "3.5.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"   % sparkVersion,
  "org.apache.spark" %% "spark-sql"    % sparkVersion,
  "org.apache.spark" %% "spark-mllib"  % sparkVersion,
  "org.apache.spark" %% "spark-graphx" % sparkVersion
)

// Run in a forked JVM so Spark can manage its own classloading
fork := true

// JVM options for Spark local mode (Java 8 compatible — no module flags needed)
javaOptions ++= Seq(
  "-Xmx4g",
  "-Xms1g"
)
