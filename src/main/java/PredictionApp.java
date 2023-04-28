
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.util.Arrays;

public class PredictionApp {

    public static void main(String[] args) {

        // Set up a SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .master("local")
                .getOrCreate();

        // Load the validation dataset
        Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .option("inferSchema", "true")
                .csv("/app/Project/src/main/resources/ValidationDataset.csv");

        // Remove double quotes from column names
        String[] currentColumns = validationData.columns();
        String[] newColumns = new String[currentColumns.length];
        for (int i = 0; i < currentColumns.length; i++) {
            newColumns[i] = currentColumns[i].replaceAll("\"", "");
        }

	// Rename the columns with new names
for (int i = 0; i < currentColumns.length; i++) {
    if (!currentColumns[i].equals(newColumns[i])) {
        validationData = validationData.withColumnRenamed(currentColumns[i], newColumns[i]);
    }
}

        // Load the trained model
        PipelineModel model = PipelineModel.load("/app/Project/classification_model");

        // Make predictions on the validation dataset
        Dataset<Row> testData = model.transform(validationData);

        // Evaluate the model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(testData);
        System.out.println("Accuracy = " + accuracy);

        // Show the predictions
        testData.select("features", "label", "prediction").show();
    }
}

