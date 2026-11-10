package heartdisease

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.DecisionTreeClassifier

object ModelTrainer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("HeartDiseaseModelTrainer")
      .master("local[*]")
      .config("spark.driver.memory", "2g")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.ui.enabled", "false")
      .getOrCreate()

    try {
      spark.sparkContext.setLogLevel("WARN")
      println("Loading heart disease dataset...")
      val raw = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("data/heart_disease_prediction.csv")

      val required = Seq(
        "id", "age", "gender", "chest_pain_type", "resting_bp",
        "cholesterol", "fasting_blood_sugar", "resting_ecg",
        "max_heart_rate", "exercise_angina", "oldpeak", "slope",
        "thal", "heart_disease"
      )
      val missing = required.filterNot(raw.columns.contains)
      require(missing.isEmpty, s"Dataset is missing columns: ${missing.mkString(", ")}")

      val categorical = Seq(
        "gender", "chest_pain_type", "resting_ecg",
        "exercise_angina", "slope", "thal"
      )
      val numeric = Seq(
        "age", "resting_bp", "cholesterol", "fasting_blood_sugar",
        "max_heart_rate", "oldpeak"
      )
      val featureCols = categorical.map(_ + "_idx") ++ numeric

      var clean = raw.na.drop(required.toArray)
      numeric.foreach(c => clean = clean.withColumn(c, col(c).cast("double")))
      clean = clean.withColumn("heart_disease", col("heart_disease").cast("double"))

      val rowCount = clean.count()
      require(rowCount > 0, "No usable records found after cleaning.")
      println(s"Training final model on $rowCount usable records.")

      val indexers = categorical.map { c =>
        new StringIndexer()
          .setInputCol(c)
          .setOutputCol(c + "_idx")
          .setHandleInvalid("keep")
      }

      val assembler = new VectorAssembler()
        .setInputCols(featureCols.toArray)
        .setOutputCol("features")

      val tree = new DecisionTreeClassifier()
        .setLabelCol("heart_disease")
        .setFeaturesCol("features")
        .setMaxDepth(10)
        .setMaxBins(32)
        .setSeed(42L)

      val pipeline = new Pipeline().setStages((indexers :+ assembler :+ tree).toArray)
      val fitted = pipeline.fit(clean)

      val output = "models/heart-disease-pipeline"
      fitted.write.overwrite().save(output)

      println(s"Saved fitted pipeline to: $output")
      println("Training complete. You can now start the prediction website.")
    } finally {
      spark.stop()
    }
  }
}
