package eg.edu.alexu.abdokhalil
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer, VectorAssembler, Word2Vec}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object App {

  def main(args : Array[String]) {
    if (args.length < 2) {
      println("Usage <input file> <algorithm>")
      println("<input file> path to a CSV file input")
      println("<algorithm> is either regression or classification")
    }
    val inputfile = args(0)
    val method = args(1)
    val conf = new SparkConf
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    println(s"Using Spark master '${conf.get("spark.master")}'")

    val spark = SparkSession
      .builder()
      .appName("Lab4")
      .config(conf)
      .getOrCreate()

    val t1 = System.nanoTime
    try {
      if (method.equals("regression"))
      {
        val input = spark.read
          .option("header",true)
          .option("delimiter",",")
          .option("inferschema",true)
          .csv(inputfile)
        val assembler = new VectorAssembler()
          .setInputCols(Array("bedrooms", "bathrooms", "sqft_living", "sqft_lot"))
          .setOutputCol("features")
        val train_test= assembler.transform(input).randomSplit(Array(0.8 , 0.2))
        val training = train_test(0)
        val test = train_test(1)
//        input.show(numRows = 5)
//        input.printSchema()
        val lr = new LinearRegression()
          .setMaxIter(10)
          .setRegParam(0.3)
          .setElasticNetParam(0.8)
          .setFeaturesCol("features")
          .setLabelCol("price")

        // Fit the model
        val lrModel = lr.fit(training)
        val trainingSummary = lrModel.summary
        trainingSummary.residuals.show()

        val predictions: DataFrame = lrModel.transform(test)
        predictions.show()
        predictions.printSchema()

        val evaluator = new RegressionEvaluator()
          .setLabelCol("price")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        val rmse: Double = evaluator.evaluate(predictions)
        println(s"RMSE is ${rmse}")
      } else if (method.equals("classification")) {
        // TODO process the sentiment data
      } else {
        println(s"Unknown algorithm ${method}")
        System.exit(1)
      }
      val t2 = System.nanoTime
      println(s"Applied algorithm $method on input $inputfile in ${(t2 - t1) * 1E-9} seconds")
    } finally {
      spark.stop
    }
  }
}
