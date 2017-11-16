package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import java.lang.Math


case class Song(year: Integer, feature1: Double, feature2: Double, feature3: Double)

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
//    val rawDF = ???

    val rdd = sc.textFile(filePath)
    
    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
      rdd.take(5).foreach(println)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(row => row.split(","))
    
    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(features => Song(features(0).substring(0,4).toInt, features(1).toDouble,
        features(2).toDouble, features(3).toDouble))
    
    //Step4: convert your rdd into a datafram
    val songsDf = songsRdd.toDF()
    
    //Cache the songs into memory
    songsRdd.cache()
    
    // Create a temp view for sql
    songsDf.createOrReplaceTempView("songs")
    
    //Number of songs
    println(songsDf.count())
    sqlContext.sql("select count(*) from from songs").show()
    
    //Number of songs released between 1998 and 2000    
    println(songsDf.filter($"year" >= 1998 && $"year" <= 2000 ).count())
    sqlContext.sql("select count(*) from songs where year >= 1998 and year <=2000").show()
    
    //Min Max Mean value of the year column
    val yearsDf = songsDf.map(song => song.getAs[Int]("year")).cache()    
    //Max     
    println(yearsDf.reduce((year1,year2) => Math.max(year1, year2))) 
    //Min
    println(yearsDf.reduce((year1,year2) => Math.min(year1, year2)))
    //Min 
    println(yearsDf.reduce((year1,year2) => year1+year2)/yearsDf.count())
    
    sqlContext.sql("select min(year), max(year), mean(year) from songs ").show()
    
    
  }
}