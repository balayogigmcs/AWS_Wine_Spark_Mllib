import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.util.Arrays;
import org.apache.spark.ml.feature.StringIndexerModel;
import java.io.IOException;
public class ClassificationModel {
    public static void main(String[] args) throws IOException {
        // Create a SparkSession
	SparkSession spark = SparkSession.builder()
                .appName("ClassificationModel")
                .master("spark://172.31.39.218:7077")
		//.master("local")
                .getOrCreate();
        // Load the data
	//Dataset<Row> data = spark.read()
        //       .option("header", "true")
        //        .option("delimiter", ";")
        //        .option("inferSchema", "true")
        //        .csv("/home/ubuntu/Project/src/main/resources/TrainingDataset.csv");
       
	Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .option("inferSchema", "true")
                .csv("/home/ubuntu/Project/src/main/resources/TrainingDataset.csv");

	// Get the current column names
String[] currentColumns = data.columns();

// Remove double quotes from column names
String[] newColumns = new String[currentColumns.length];
for (int i = 0; i < currentColumns.length; i++) {
    newColumns[i] = currentColumns[i].replaceAll("\"", "");
}

// Rename the columns with new names
for (int i = 0; i < currentColumns.length; i++) {
    if (!currentColumns[i].equals(newColumns[i])) {
        data = data.withColumnRenamed(currentColumns[i], newColumns[i]);
    }
}

	// Prepare the data for training
        String[] inputCols = data.columns();
	for(int i=0; i < inputCols.length ; i++)
	{
    		inputCols[i] = inputCols[i].replaceAll("\"", "");
	}
        inputCols = Arrays.copyOf(inputCols, inputCols.length - 1);

        
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("quality")
                .setOutputCol("label")
                .fit(data);

        // Train the classification model
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.01)
                .setLabelCol("label")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {assembler, labelIndexer, lr});

        PipelineModel model = pipeline.fit(data);

        // Evaluate the model
        Dataset<Row> predictions = model.transform(data);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);

        // Print the model's accuracy
        System.out.println("Accuracy = " + accuracy);

        // Save the model
        model.write().overwrite().save("classification_model_parallel");
    }
}

