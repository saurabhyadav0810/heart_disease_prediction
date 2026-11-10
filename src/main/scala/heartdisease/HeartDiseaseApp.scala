package heartdisease

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.graphx.{Edge, Graph}
import org.apache.spark.rdd.RDD

object HeartDiseaseApp {

  private val DataPath = "data/heart_disease_prediction.csv"
  private val TargetCol = "heart_disease"
  private val IdCol = "id"

  private val categoricalCols = Array(
    "gender",
    "chest_pain_type",
    "resting_ecg",
    "exercise_angina",
    "slope",
    "thal"
  )

  private val numericCols = Array(
    "age",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "max_heart_rate",
    "oldpeak"
  )

  private val featureCols = numericCols ++ categoricalCols.map(_ + "_indexed")

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Heart Disease Prediction and Healthcare Analytics")
      .master("local[*]")
      .config("spark.sql.shuffle.partitions", "8")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try {
      println("\n" + "=" * 78)
      println(" HEART DISEASE PREDICTION AND HEALTHCARE ANALYTICS")
      println("=" * 78)

      val rawDF = loadData(spark)
      val cleanDF = cleanData(rawDF)

      performEDA(cleanDF)
      val preparedDF = prepareFeatures(cleanDF)
      trainAndEvaluateModel(preparedDF)
      performGraphXAnalysis(cleanDF)
      runPerformanceExperiment(cleanDF)

      println("\n" + "=" * 78)
      println(" APPLICATION COMPLETED SUCCESSFULLY")
      println("=" * 78)
    } finally {
      spark.stop()
    }
  }

  private def time[T](label: String)(block: => T): T = {
    val start = System.nanoTime()
    val result = block
    val elapsed = (System.nanoTime() - start) / 1e9
    println(f"$label completed in $elapsed%.3f seconds")
    result
  }

  private def section(title: String): Unit = {
    println("\n" + "-" * 78)
    println(s" $title")
    println("-" * 78)
  }

  // 1. Load the CSV dataset.
  private def loadData(spark: SparkSession): DataFrame = {
    section("1. DATA LOADING")

    val df = time("CSV loading") {
      spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("mode", "DROPMALFORMED")
        .csv(DataPath)
    }

    println(s"Dataset path : $DataPath")
    println(s"Rows         : ${df.count()}")
    println(s"Columns      : ${df.columns.length}")
    println("Schema:")
    df.printSchema()
    df.show(5, truncate = false)
    df
  }

  // 2. Basic data quality checks and cleaning.
  private def cleanData(df: DataFrame): DataFrame = {
    section("2. DATA CLEANING")

    println("Missing values by column:")
    val missingExpressions = df.columns.map { c =>
      sum(
        when(
          col(c).isNull || trim(col(c).cast(StringType)) === "",
          1
        ).otherwise(0)
      ).alias(c)
    }

    df.select(missingExpressions: _*).show(truncate = false)

    val before = df.count()
    val noNulls = df.na.drop(df.columns)

    val noDuplicates =
      if (df.columns.contains(IdCol)) noNulls.dropDuplicates(IdCol)
      else noNulls.dropDuplicates()

    val after = noDuplicates.count()

    println(s"Rows before cleaning : $before")
    println(s"Rows after cleaning  : $after")
    println(s"Rows removed         : ${before - after}")

    val cleaned = noDuplicates.withColumn(
      TargetCol,
      col(TargetCol).cast(DoubleType)
    )

    println("Target class distribution:")
    cleaned.groupBy(TargetCol).count().orderBy(TargetCol).show()

    cleaned
  }

  // 3. Exploratory Data Analysis.
  private def performEDA(df: DataFrame): Unit = {
    section("3. EXPLORATORY DATA ANALYSIS")

    println("Descriptive statistics for numerical columns:")
    df.select(numericCols.map(col): _*)
      .describe()
      .show(truncate = false)

    println("Heart disease class distribution:")

    val total = df.count().toDouble

    df.groupBy(TargetCol)
      .count()
      .orderBy(TargetCol)
      .show()

    df.groupBy(TargetCol)
      .count()
      .collect()
      .foreach { row =>
        val label = row.getAs[Double](TargetCol)
        val count = row.getAs[Long]("count")

        println(
          f"Class $label%.0f: $count%,d records (${count / total * 100}%.2f%%)"
        )
      }

    println("Gender-wise heart disease distribution:")

    df.groupBy("gender", TargetCol)
      .count()
      .orderBy("gender", TargetCol)
      .show(30, truncate = false)

    println("Average clinical measurements by heart disease status:")

    df.groupBy(TargetCol)
      .agg(
        round(avg("age"), 2).alias("avg_age"),
        round(avg("resting_bp"), 2).alias("avg_resting_bp"),
        round(avg("cholesterol"), 2).alias("avg_cholesterol"),
        round(avg("max_heart_rate"), 2).alias("avg_max_heart_rate"),
        round(avg("oldpeak"), 2).alias("avg_oldpeak")
      )
      .orderBy(TargetCol)
      .show(truncate = false)

    println("Pearson correlations with heart_disease:")

    numericCols.foreach { c =>
      val correlation = df.stat.corr(c, TargetCol)
      println(f"$c%-22s : $correlation%.4f")
    }

    df.createOrReplaceTempView("heart_patients")

    println("Spark SQL: number of records in each target class")
    sparkSqlClassSummary(df)
  }

  private def sparkSqlClassSummary(df: DataFrame): Unit = {
    val result = df.sparkSession.sql(
      """
        |SELECT heart_disease,
        |       COUNT(*) AS patient_count,
        |       ROUND(AVG(age), 2) AS average_age,
        |       ROUND(AVG(cholesterol), 2) AS average_cholesterol
        |FROM heart_patients
        |GROUP BY heart_disease
        |ORDER BY heart_disease
        |""".stripMargin
    )

    result.show(truncate = false)
  }

  // 4. Feature engineering.
  private def prepareFeatures(df: DataFrame): DataFrame = {
    section("4. FEATURE ENGINEERING")

    val indexers = categoricalCols.map { c =>
      new StringIndexer()
        .setInputCol(c)
        .setOutputCol(c + "_indexed")
        .setHandleInvalid("keep")
    }

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val pipeline = new Pipeline()
      .setStages(indexers :+ assembler)

    val transformed = time("Feature transformation") {
      pipeline.fit(df).transform(df)
    }

    println(s"Input feature count: ${featureCols.length}")

    transformed
      .select("features", TargetCol)
      .show(5, truncate = false)

    transformed
  }

  // 5. Decision Tree classification and evaluation.
  private def trainAndEvaluateModel(df: DataFrame): Unit = {
    section("5. MACHINE LEARNING - DECISION TREE CLASSIFIER")

    val Array(trainingData, testData) =
      df.randomSplit(Array(0.8, 0.2), seed = 42L)

    println(s"Training records: ${trainingData.count()}")
    println(s"Testing records : ${testData.count()}")

    val classifier = new DecisionTreeClassifier()
      .setLabelCol(TargetCol)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxDepth(10)
      .setMaxBins(32)
      .setSeed(42L)

    val model = time("Decision Tree training") {
      classifier.fit(trainingData)
    }

    val predictions = time("Test-set prediction") {
      model.transform(testData).cache()
    }

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(TargetCol)
      .setPredictionCol("prediction")

    val accuracy = evaluator
      .setMetricName("accuracy")
      .evaluate(predictions)

    val weightedPrecision = evaluator
      .setMetricName("weightedPrecision")
      .evaluate(predictions)

    val weightedRecall = evaluator
      .setMetricName("weightedRecall")
      .evaluate(predictions)

    val weightedF1 = evaluator
      .setMetricName("f1")
      .evaluate(predictions)

    println("\nOverall model metrics:")
    println(f"Accuracy           : ${accuracy * 100}%.2f%%")
    println(f"Weighted precision : $weightedPrecision%.4f")
    println(f"Weighted recall    : $weightedRecall%.4f")
    println(f"Weighted F1-score  : $weightedF1%.4f")

    val counts = predictions
      .select(
        col(TargetCol).cast(IntegerType).alias("actual"),
        col("prediction").cast(IntegerType).alias("predicted")
      )
      .groupBy("actual", "predicted")
      .count()
      .collect()
      .map(r => ((r.getInt(0), r.getInt(1)), r.getLong(2)))
      .toMap

    val tn = counts.getOrElse((0, 0), 0L).toDouble
    val fp = counts.getOrElse((0, 1), 0L).toDouble
    val fn = counts.getOrElse((1, 0), 0L).toDouble
    val tp = counts.getOrElse((1, 1), 0L).toDouble

    val precision = safeDivide(tp, tp + fp)
    val recall = safeDivide(tp, tp + fn)
    val specificity = safeDivide(tn, tn + fp)
    val f1 = safeDivide(
      2 * precision * recall,
      precision + recall
    )
    val balancedAccuracy = (recall + specificity) / 2.0

    println("\nConfusion matrix:")
    println("                     Predicted 0     Predicted 1")
    println(f"Actual 0             $tn%12.0f     $fp%12.0f")
    println(f"Actual 1             $fn%12.0f     $tp%12.0f")

    println("\nClass-1 and binary classification metrics:")
    println(f"Precision            : $precision%.4f")
    println(f"Recall / sensitivity : $recall%.4f")
    println(f"Specificity          : $specificity%.4f")
    println(f"F1-score             : $f1%.4f")
    println(f"Balanced accuracy    : $balancedAccuracy%.4f")

    println("\nDecision Tree feature importances:")

    val importanceRows =
      featureCols.zip(model.featureImportances.toArray)
        .sortBy(-_._2)

    importanceRows.foreach { case (name, importance) =>
      println(f"$name%-28s ${importance}%.4f")
    }

    predictions.unpersist()
  }

  private def safeDivide(
      numerator: Double,
      denominator: Double
  ): Double = {
    if (denominator == 0.0) 0.0
    else numerator / denominator
  }

  // 6. GraphX analysis.
  // Patient-to-category links represent attributes, not real interactions.
  private def performGraphXAnalysis(df: DataFrame): Unit = {
    section("6. GRAPH ANALYTICS USING GRAPHX")

    // Corrected Scala varargs syntax.
    val graphColumns =
      Array(col(IdCol).cast(LongType).alias("patient_id")) ++
        categoricalCols.map(col)

    val sample = df.select(graphColumns: _*)
      .sample(
        withReplacement = false,
        fraction = 0.10,
        seed = 42L
      )
      .na.drop()
      .cache()

    val patientRows = sample.collect()

    val patientVertices: RDD[(Long, String)] =
      df.sparkSession.sparkContext.parallelize(
        patientRows.map { row =>
          (row.getAs[Long]("patient_id"), "Patient")
        }
      )

    val categoryValues = categoricalCols.flatMap { c =>
      patientRows.map { row =>
        s"$c=${row.getAs[Any](c)}"
      }
    }.distinct.sorted

    val categoryIdByName = categoryValues.zipWithIndex.map {
      case (name, index) =>
        name -> (Long.MinValue + index.toLong)
    }.toMap

    val categoryVertices: RDD[(Long, String)] =
      df.sparkSession.sparkContext.parallelize(
        categoryIdByName.toSeq.map {
          case (name, id) => (id, name)
        }
      )

    val vertices = patientVertices.union(categoryVertices)

    val edgeData = patientRows.flatMap { row =>
      val patientId = row.getAs[Long]("patient_id")

      categoricalCols.map { c =>
        val categoryName = s"$c=${row.getAs[Any](c)}"

        Edge(
          patientId,
          categoryIdByName(categoryName),
          1L
        )
      }
    }

    val edges: RDD[Edge[Long]] =
      df.sparkSession.sparkContext.parallelize(edgeData)

    val graph = Graph(vertices, edges)

    val vertexCount = graph.vertices.count()
    val edgeCount = graph.edges.count()

    val componentCount = graph.connectedComponents()
      .vertices
      .map(_._2)
      .distinct()
      .count()

    println(s"Patient vertices          : ${patientVertices.count()}")
    println(s"Clinical category vertices: ${categoryVertices.count()}")
    println(s"Total graph vertices      : $vertexCount")
    println(s"Total graph edges         : $edgeCount")
    println(s"Connected components      : $componentCount")

    println(
      "Graph interpretation: patient records are connected to their"
    )
    println(
      "categorical clinical attribute values in this sampled dataset."
    )

    sample.unpersist()
  }

  // 7. Performance experiment.
  private def runPerformanceExperiment(df: DataFrame): Unit = {
    section("7. PERFORMANCE EXPERIMENT")

    println(
      "Comparing Decision Tree training time at different data sizes."
    )

    println(
      "This is a local-machine experiment, not a multi-node cluster benchmark."
    )

    val fractions = Array(0.2, 0.4, 0.6, 0.8, 1.0)

    println("\nData fraction      Rows           Training time (s)")
    println("-" * 55)

    fractions.foreach { fraction =>
      val sample = df.sample(
        withReplacement = false,
        fraction = fraction,
        seed = 42L
      ).cache()

      val rowCount = sample.count()

      val indexers = categoricalCols.map { c =>
        new StringIndexer()
          .setInputCol(c)
          .setOutputCol(c + "_indexed")
          .setHandleInvalid("keep")
      }

      val assembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("features")
        .setHandleInvalid("keep")

      val featurePipeline = new Pipeline()
        .setStages(indexers :+ assembler)

      val featured = featurePipeline.fit(sample).transform(sample)

      val classifier = new DecisionTreeClassifier()
        .setLabelCol(TargetCol)
        .setFeaturesCol("features")
        .setMaxDepth(10)
        .setMaxBins(32)
        .setSeed(42L)

      val start = System.nanoTime()

      classifier.fit(featured)

      val seconds = (System.nanoTime() - start) / 1e9

      val fractionLabel = f"${fraction * 100}%.0f%%"

      println(
        f"$fractionLabel%-18s $rowCount%-14d $seconds%.3f"
      )

      sample.unpersist()
    }
  }
}