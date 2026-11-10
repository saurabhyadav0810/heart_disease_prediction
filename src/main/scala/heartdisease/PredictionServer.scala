package heartdisease

import java.io.ByteArrayOutputStream
import java.net.{InetSocketAddress, URLDecoder}
import java.util.concurrent.Executors
import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vector

object PredictionServer {
  private val modelPath = "models/heart-disease-pipeline"

  private val inputSchema = StructType(Seq(
    StructField("gender", StringType, nullable = false),
    StructField("chest_pain_type", StringType, nullable = false),
    StructField("resting_ecg", StringType, nullable = false),
    StructField("exercise_angina", StringType, nullable = false),
    StructField("slope", StringType, nullable = false),
    StructField("thal", StringType, nullable = false),
    StructField("age", DoubleType, nullable = false),
    StructField("resting_bp", DoubleType, nullable = false),
    StructField("cholesterol", DoubleType, nullable = false),
    StructField("fasting_blood_sugar", DoubleType, nullable = false),
    StructField("max_heart_rate", DoubleType, nullable = false),
    StructField("oldpeak", DoubleType, nullable = false)
  ))

  private def readBody(exchange: HttpExchange): String = {
    val in = exchange.getRequestBody
    val out = new ByteArrayOutputStream()
    val buffer = new Array[Byte](4096)
    var n = in.read(buffer)
    while (n != -1) {
      out.write(buffer, 0, n)
      n = in.read(buffer)
    }
    new String(out.toByteArray, "UTF-8")
  }

  private def parseForm(body: String): Map[String, String] = {
    body.split("&").filter(_.nonEmpty).map { part =>
      val pair = part.split("=", 2)
      val key = URLDecoder.decode(pair(0), "UTF-8")
      val value = if (pair.length > 1) URLDecoder.decode(pair(1), "UTF-8") else ""
      key -> value
    }.toMap
  }

  private def send(exchange: HttpExchange, status: Int, contentType: String, body: Array[Byte]): Unit = {
    exchange.getResponseHeaders.set("Content-Type", contentType)
    exchange.getResponseHeaders.set("Cache-Control", "no-store")
    exchange.sendResponseHeaders(status, body.length)
    val os = exchange.getResponseBody
    os.write(body)
    os.close()
  }

  private def sendText(exchange: HttpExchange, status: Int, contentType: String, body: String): Unit =
    send(exchange, status, contentType, body.getBytes("UTF-8"))

  private def resource(path: String): Option[Array[Byte]] = {
    val stream = getClass.getClassLoader.getResourceAsStream("web/" + path)
    if (stream == null) None
    else {
      val out = new ByteArrayOutputStream()
      val buffer = new Array[Byte](4096)
      var n = stream.read(buffer)
      while (n != -1) {
        out.write(buffer, 0, n)
        n = stream.read(buffer)
      }
      stream.close()
      Some(out.toByteArray)
    }
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("HeartDiseasePredictionWeb")
      .master("local[*]")
      .config("spark.driver.memory", "2g")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.ui.enabled", "false")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    try {
      val model = PipelineModel.load(modelPath)
      println(s"Loaded saved model from $modelPath")

      val server = HttpServer.create(new InetSocketAddress("127.0.0.1", 8080), 0)
      server.setExecutor(Executors.newFixedThreadPool(4))

      server.createContext("/", new HttpHandler {
        override def handle(exchange: HttpExchange): Unit = {
          val path = exchange.getRequestURI.getPath
          if (path == "/api/health" && exchange.getRequestMethod == "GET") {
            sendText(exchange, 200, "application/json; charset=utf-8", """{"status":"ready"}""")
          } else if (exchange.getRequestMethod == "GET") {
            val file = if (path == "/") "index.html" else path.stripPrefix("/")
            val contentType =
              if (file.endsWith(".css")) "text/css; charset=utf-8"
              else if (file.endsWith(".js")) "application/javascript; charset=utf-8"
              else "text/html; charset=utf-8"
            resource(file) match {
              case Some(bytes) => send(exchange, 200, contentType, bytes)
              case None => sendText(exchange, 404, "text/plain; charset=utf-8", "Not found")
            }
          } else if (path == "/api/predict" && exchange.getRequestMethod == "POST") {
            try {
              val p = parseForm(readBody(exchange))
              def required(name: String): String =
                p.getOrElse(name, throw new IllegalArgumentException("Missing field: " + name))
              def number(name: String): Double = {
                val value = required(name).toDouble
                if (java.lang.Double.isNaN(value) || java.lang.Double.isInfinite(value)) throw new IllegalArgumentException("Invalid number: " + name)
                value
              }

              val row = Row(
                required("gender"),
                required("chest_pain_type"),
                required("resting_ecg"),
                required("exercise_angina"),
                required("slope"),
                required("thal"),
                number("age"),
                number("resting_bp"),
                number("cholesterol"),
                number("fasting_blood_sugar"),
                number("max_heart_rate"),
                number("oldpeak")
              )

              val input = spark.createDataFrame(
                spark.sparkContext.parallelize(Seq(row)), inputSchema
              )
              val result = model.transform(input).select("prediction", "probability").first()
              val prediction = result.getAs[Double]("prediction").toInt
              val probabilities = result.getAs[Vector]("probability").toArray
              val positiveProbability = if (probabilities.length > 1) probabilities(1) else 0.0
              val json =
                s"""{"prediction":$prediction,"positiveProbability":${positiveProbability}}"""
              sendText(exchange, 200, "application/json; charset=utf-8", json)
            } catch {
              case e: Exception =>
                val safe = Option(e.getMessage).getOrElse("Invalid input").replace("\\", "\\\\").replace("\"", "\\\"")
                sendText(exchange, 400, "application/json; charset=utf-8", s"""{"error":"$safe"}""")
            }
          } else {
            sendText(exchange, 404, "text/plain; charset=utf-8", "Not found")
          }
        }
      })

      server.start()
      println("HeartCare Analytics is running at http://localhost:8080")
      println("Keep this terminal open while using the website. Press Ctrl+C to stop.")
      while (true) Thread.sleep(60000)
    } finally {
      spark.stop()
    }
  }
}
