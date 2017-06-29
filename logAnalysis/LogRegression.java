package org.deeplearning4j.logAnalysis;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by Liu
 */
public class LogRegression {
    //Random number generator seed, for reproducability
    public static final int seed = 12345;
    //Number of iterations per minibatch
    public static final int iterations = 1;
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 1;
    //Number of data points
//    public static final int nSamples = 10000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 128;
    //Network learning rate
    public static final double learningRate = 0.001;

    public static final Random rng = new Random(seed);

    public static void main(String[] args){

        //Generate the training data
        DataSetIterator iterator = getTrainingData(batchSize,rng);

        //Create the network
        int numInput = 3;
        int numOutputs = 20;
        int nHidden = 16;
        
        FileOutputStream fos = null;
        String[] waveletArr={ "Haar","Daubechies 2","Daubechies 3","Daubechies 4","Daubechies 5","Daubechies 6","Daubechies 7","Daubechies 8","Daubechies 9","Daubechies 10","Daubechies 11","Daubechies 12","Daubechies 13","Daubechies 14","Daubechies 15","Daubechies 16","Daubechies 17","Daubechies 18","Daubechies 19","Daubechies 20","Coiflet 1","Coiflet 2","Coiflet 3","Coiflet 4","Coiflet 5","Symlet 2","Symlet 3","Symlet 4","Symlet 5","Symlet 6","Symlet 7","Symlet 8","Symlet 9","Symlet 10","Symlet 11","Symlet 12","Symlet 13","Symlet 14","Symlet 15","Symlet 16","Symlet 17","Symlet 18","Symlet 19","Symlet 20","BiOrthogonal 1/1","BiOrthogonal 1/3","BiOrthogonal 1/5","BiOrthogonal 3/1","BiOrthogonal 3/3","BiOrthogonal 3/5","BiOrthogonal 3/7","BiOrthogonal 3/9"};
        try {
			fos=new FileOutputStream(new File("d:/LogAnalysis.txt"));
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
        double maxAccuracy=0;
        
        for(String waveletIdentifier : waveletArr)
        {
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
/*                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                        .activation("tanh")
                        .build())*/
                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                        .activation(new WaveletActivation(waveletIdentifier))
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nIn(nHidden).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(1));


        //Train the network on the full data set, and evaluate in periodically
        iterator.reset();
        for( int i=0; i<nEpochs; i++ ){
            net.fit(iterator);
        }    
        DataSetIterator testIterator = getTestingData(batchSize,rng);        		
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        //model.output(testIter);
        while(testIterator.hasNext()){
            DataSet t = testIterator.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = net.output(features,false);
            eval.eval(lables, predicted);
        }

        //Print the evaluation statistics
        try {
			fos.write((waveletIdentifier+ " Start Wavelet\r\n").getBytes());
			fos.write((eval.stats()+"\r\n").getBytes());
			fos.write((waveletIdentifier+"\r\n").getBytes());
			fos.write((waveletIdentifier+ " End Wavelet\r\n").getBytes());
			maxAccuracy=Math.max(maxAccuracy, eval.accuracy());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
        
        System.out.println(eval.stats());
        }
        
        try {
			fos.write(("The maximun value of Arrucracy is: "+maxAccuracy).getBytes());
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
    	if(fos!=null)
    	{
    		try {
    			fos.close();
    		} catch (IOException e) {
    			e.printStackTrace();
    		}
    	}
    }

    private static DataSetIterator getTestingData(int batchSize, Random rand){
        int numOutputs = 20;
        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        DataSetIterator testIter = null;
        try{
        rr.initialize(new FileSplit(new File("src/main/resources/logAnalysis/testingRegression.csv")));
        testIter = new RecordReaderDataSetIterator(rr,batchSize,3,numOutputs);
        }catch(Exception e){
        }
        return testIter;
    }
    
    private static DataSetIterator getTrainingData(int batchSize, Random rand){	
        int numOutputs = 20;
        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        DataSetIterator trainIter = null;
        try{
        rr.initialize(new FileSplit(new File("src/main/resources/logAnalysis/trainingRegression.csv")));
        trainIter = new RecordReaderDataSetIterator(rr,batchSize,3,numOutputs);
        }catch(Exception e){
        	
        }
        return trainIter;
    }
}
