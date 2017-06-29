package org.deeplearning4j.logAnalysis.lstm.uci;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.logAnalysis.WaveletActivation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class AirQualityLstmPredict {
	private static final int IN_NUM = 1;
	private static final int OUT_NUM = 1;
	private static final int Epochs = 180;
	  
	private static final int lstmLayer1Size = 4;
//	private static final int lstmLayer2Size = 4;
	
	private static final int predictSteps = 1;
	  
	public static MultiLayerNetwork getNetModel(int nIn,int nOut, String waveletID){
	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
	        .learningRate(0.001) //0.1
	        .rmsDecay(0.85)
	        .seed(12345)
	        .regularization(true)
	        .l2(0.001)
	        .weightInit(WeightInit.XAVIER)
	        .updater(Updater.NESTEROVS).momentum(0.9)	//RMSPROP
	        .list()
	        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayer1Size)  
	            .activation(Activation.TANH).build())	//.activation(new WaveletActivation(waveletID)).build())
//	        .layer(1, new GravesLSTM.Builder().nIn(lstmLayer1Size).nOut(lstmLayer2Size)
//	            .activation(Activation.TANH).build())
	        .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
	            .nIn(lstmLayer1Size).nOut(nOut).build())
	        .pretrain(false).backprop(true)
	        .backpropType(BackpropType.TruncatedBPTT)
//	        .tBPTTForwardLength(50) //50-200
//	        .tBPTTBackwardLength(50)
	        .build();
	  
	    MultiLayerNetwork net = new MultiLayerNetwork(conf);  
	    net.init();  
	    net.setListeners(new ScoreIterationListener(1));  //set monitor, print score every iteration step
	  
	    return net;  
	}
	
	public static void train(MultiLayerNetwork net,AirQualityDataIterator iterator, INDArray[] initArrayArr, FileOutputStream fos){
		//Iteration training
		for(int i=0;i<Epochs;i++) {
			DataSet dataSet = null;
			while (iterator.hasNext()) {
				dataSet = iterator.next();
				net.fit(dataSet);
			}
			iterator.reset();
			// record log
			try {
				fos.write(("Finished the number "+(i+1)+" round training!!\r\n").getBytes());
				test(net, iterator, initArrayArr, fos);//////////
			} catch (IOException e) {
				e.printStackTrace();
			}
			net.rnnClearPreviousState();
		}
	}
	
	public static void test(MultiLayerNetwork net,AirQualityDataIterator iterator, INDArray[] initArrayArr, FileOutputStream fos) throws IOException
	{
		for(int i=0; i<initArrayArr.length; i++)
		{
		INDArray initArray=initArrayArr[i];
		INDArray output = net.rnnTimeStep(initArray);
		//record log
		fos.write(("For tesing item "+(i+1)+" , The predicted result is: \r\n").getBytes());
		
		float[] maxNums = iterator.getMaxArr();
		for(int j=0;j<predictSteps;j++) {
			fos.write((output.getDouble(0)*maxNums[0]+"\r\n").getBytes());
            output = net.rnnTimeStep(initArray);//Use the last output as input, run a new time step
		}
		}
	}
	
	private static INDArray[] getTestingArray(AirQualityDataIterator iter) throws IOException{
		int TESTING_ITEM_SIZE=5; // the number of data items to pick for predicting
		int testItems=TESTING_ITEM_SIZE; //Read the number of items from testing data set
        List<Float> dataList = new ArrayList<>();
        String inputFile = AirQualityLstmPredict.class.getClassLoader().getResource("logAnalysis/UCI/AirQuality/AirQualityUCI_CO.csv").getPath();
        FileInputStream fis = new FileInputStream(inputFile);
        BufferedReader in = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
        in.readLine(); //skip the first line (title)
        String line = in.readLine();
        System.out.println("Reading verifying data items..");
        while(line!=null && testItems-->0){
            dataList.add(Float.parseFloat(line));
            line = in.readLine();
        }
        in.close();
        fis.close();
        // instantiate each element to INDArray
        INDArray[] indArr=new INDArray[TESTING_ITEM_SIZE];
        for(int i=0; i<indArr.length; i++)
        {
        	float[] maxNums = iter.getMaxArr();
        	indArr[i] = Nd4j.zeros(1, 1, 1);
        	indArr[i].putScalar(new int[]{0,0,0}, (double)dataList.get(i)/maxNums[0]);
        }
		return indArr;
	}
	
	public static void main(String[] args) {
	    String inputFile = AirQualityLstmPredict.class.getClassLoader().getResource("logAnalysis/UCI/AirQuality/AirQualityUCI_CO.csv").getPath();
	    int batchSize = 1;
	    int exampleLength = 2;
	    //Initialize the Neural Network
	    AirQualityDataIterator iterator = new AirQualityDataIterator();
	    iterator.loadData(inputFile,batchSize,exampleLength);
	  
	    try {
			INDArray[] initArrayArr = getTestingArray(iterator);
			FileOutputStream fos=new FileOutputStream(new File("d:/AirQualityUCI_CO.txt"));
			
			boolean USE_WAVELET=false;
			
			if(USE_WAVELET)
			{
				// Add wavelets to model
				String[] waveletArr={ "Haar","Daubechies 2","Daubechies 3","Daubechies 4","Daubechies 5","Daubechies 6","Daubechies 7","Daubechies 8","Daubechies 9","Daubechies 10","Daubechies 11","Daubechies 12","Daubechies 13","Daubechies 14","Daubechies 15","Daubechies 16","Daubechies 17","Daubechies 18","Daubechies 19","Daubechies 20","Coiflet 1","Coiflet 2","Coiflet 3","Coiflet 4","Coiflet 5","Symlet 2","Symlet 3","Symlet 4","Symlet 5","Symlet 6","Symlet 7","Symlet 8","Symlet 9","Symlet 10","Symlet 11","Symlet 12","Symlet 13","Symlet 14","Symlet 15","Symlet 16","Symlet 17","Symlet 18","Symlet 19","Symlet 20","BiOrthogonal 1/1","BiOrthogonal 1/3","BiOrthogonal 1/5","BiOrthogonal 3/1","BiOrthogonal 3/3","BiOrthogonal 3/5","BiOrthogonal 3/7","BiOrthogonal 3/9"};
				for(String waveletID : waveletArr)
				{
					fos.write(("Begin the network model of: "+waveletID+"\r\n").getBytes());
					MultiLayerNetwork net = getNetModel(IN_NUM,OUT_NUM, waveletID);
					train(net, iterator, initArrayArr, fos);
					fos.write(("End the network model of: "+waveletID+"\r\n").getBytes());
				}
			}else{
				MultiLayerNetwork net = getNetModel(IN_NUM,OUT_NUM, "");
				train(net, iterator, initArrayArr, fos);
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 
}
