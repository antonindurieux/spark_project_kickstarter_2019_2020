package paristech

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._


object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Limitation des messages de la console au niveau "WARN"
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    // Import des fonctions implicites
    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("\n")
    println("Preprocessor launched")
    println("\n")

    // Chargement des données dans un dataframe
    val df: DataFrame = spark
      .read
      .option("header", value = true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("data/train_clean.csv")

    // Affichage du nombre de lignes et de colonnes dans le DataFrame
    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    // Affichage d'un extrait du dataframe
    df.show()

    // Affichage du schéma du dataframe
    df.printSchema()

    // Typage des colonnes contenant des entiers
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    // Description statistique des colonnes goal, backers_count et final_status
    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    // Affichage de quelques caractéristiques du jeu de donnée
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    // Suppression de la colonne disable_communication
    val df2: DataFrame = dfCasted.drop("disable_communication")

    // Suppression des données du futur
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    // Création des UDF cleanCountryUdf et cleanCurrencyUdf
    // fonction cleanCountry
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False" || country == "True")
        currency
      else if (country != null && country.length != 2)
        null
      else
        country
    }

    // fonction cleanCurrency
    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    // Création du dataframe avec les colonnes country2 et currency2, nettoyées
    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    // Suppression des lignes dont la valeur de final_status est incorrecte
    val dfFinStatusClean: DataFrame = dfCountry.filter($"final_status" === 0 || $"final_status" === 1)

    // Création d'une colonne days_campaign, nombre de jours entre launched_at et deadline
    val dfDaysCampaign: DataFrame = dfFinStatusClean
      .withColumn("launched_at_date", from_unixtime($"launched_at"))
      .withColumn("deadline_date", from_unixtime($"deadline"))
      .withColumn("days_campaign", datediff($"deadline_date", $"launched_at_date"))

    // Création d'une colonne hours_prepa, nombre d'heures entre launched_at et created_at
    val dfHoursPrepa: DataFrame = dfDaysCampaign
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at") / 3600, 3))
      .drop("launched_at", "created_at", "deadline")

    // Conversion des colonnes name, desc et keywords en minuscule
    val dfLowerCase: DataFrame = dfHoursPrepa
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    // Concaténation des colonnes name, desc et keywords
    val dftextConcat: DataFrame = dfLowerCase
      .withColumn("text", concat($"name", lit(" "), $"desc", lit(" "), $"keywords"))
      .drop("name", "desc", "keywords")

    // Remplacement des null dans les colonnes days_campaign, hours_prepa, goal, country2 et currency2
    val dfNoNull: DataFrame = dftextConcat
      .na.fill(-1, Seq("days_campaign", "hours_prepa", "goal"))
      .na.fill("unknown", Seq("country2", "currency2"))

    // Ajout personnel: Suppression des heures de préparation négative
    val dfHoursClean: DataFrame = dfNoNull.filter($"hours_prepa" > 0)

    // Sauvegarde au format parquet (avec écrasement si il existe déjà)
    dfHoursClean.write.mode(SaveMode.Overwrite).parquet("data/export_parquet/")

    println("\n")
    println("Preprocessing done!")
    println("\n")

  }
}
