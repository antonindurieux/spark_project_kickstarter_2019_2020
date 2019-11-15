package paristech

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession



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

    // DataFrame loading
    val data = spark.read.parquet("/home/antonin/Dropbox/Cours_Paristech/INF729-Introduction_Hadoop/Spark/cours-spark-telecom-master/data/prepared_trainingset/*.parquet")

    // Training/test set splitting
    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))

    println("Pipeline building...")

    // Stage 1 : Récupération des mots des textes
    val text_tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Stage 2 : Suppression des stop words
    val stopword_remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // Stage 3 : Calcul de la partie TF (transformation des mots en vecteurs de comptage)
    val wordtoken_vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")

    // Stage 4 : Calcul de la partie IDF
    val wordvector_idf = new IDF()
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
    val onehot_encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    // Stage 9 : Assemblage des features en un unique vecteur
    val feature_assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // Stage 10 : Création et instanciation du modèle de classification
    val logistic_reg = new LogisticRegression()
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
      .setStages(Array(text_tokenizer, stopword_remover, wordtoken_vectorizer, wordvector_idf,
        country_indexer, currency_indexer, onehot_encoder, feature_assembler, logistic_reg))

    // Parameters grid to search over
    val paramGrid = new ParamGridBuilder()
      .addGrid(logistic_reg.elasticNetParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(wordtoken_vectorizer.minDF, Array(55.0, 75.0, 95.0))
      .build()

    // Performance evaluation
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // Train-validation split
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setParallelism(2)

    // Model fitting
    println("Model fitting...")
    val model = trainValidationSplit.fit(trainingData)

    // Predictions on test data
    println("Computing predictions on test data...")
    val dfWithPredictions = model.transform(testData)
    dfWithPredictions.groupBy("final_status", "predictions").count.show()

    val f1_score = evaluator.evaluate(dfWithPredictions)
    println(s"F1 Score = $f1_score")

  }
}
