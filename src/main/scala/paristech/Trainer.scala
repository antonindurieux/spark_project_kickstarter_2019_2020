package paristech

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator



object Trainer {

  def main(args: Array[String]): Unit = {

    // Limitation des messages de la console au niveau "WARN"
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("Trainer launched")

    // Chargement du DataFrame
    val data = spark.read.parquet("/home/antonin/Dropbox/Cours_Paristech/INF729-Introduction_Hadoop/Spark/cours-spark-telecom-master/data/prepared_trainingset/*.parquet")

    // Split en training/test set
    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))

    println("Pipeline building...")

    // Stage 1 : Récupération des mots des textes
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Stage 2 : Suppression des stop words
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // Stage 3 : Calcul de la partie TF (transformation des mots en vecteurs de comptage)
    val cv_text = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")

    // Stage 4 : Calcul de la partie IDF
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("tfidf")

    // Stage 5 : Conversion de country2 en quantité numérique
    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    // Stage 6 : Conversion de currency2 en quantité numérique
    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")

    // Stage 7/8 : One-Hot encoding de country2 et currency2
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    // Stage 9 : Assemblage des features en un unique vecteur
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // Stage 10 : Création et instanciation du modèle de classification
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    // Création du pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cv_text, idf, country_indexer, currency_indexer, encoder, assembler, lr))

    println("Model fitting...")

    // Entrainement du modèle
    val model = pipeline.fit(trainingData)

    println("Prediction on test data...")

    // Prédictions sur les données test
    val dfWithSimplePredictions = model.transform(testData)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")
    val f1_score = evaluator.evaluate(dfWithSimplePredictions)
    println(s"F1 Score = $f1_score")

  }
}
